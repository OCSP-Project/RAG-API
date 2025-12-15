import re
import logging
from typing import Tuple, Optional, List, Dict, Any
from app.models.schemas import ChatIntent, ContractorAction
from app.config.settings import settings

logger = logging.getLogger("rag-api")

# Compile regex patterns once for performance
BUDGET_PATTERNS = [
    re.compile(r'(\d+)\s*tỷ'),
    re.compile(r'dưới\s*(\d+)\s*tỷ'),
    re.compile(r'khoảng\s*(\d+)\s*tỷ'),
    re.compile(r'(\d+)\s*-\s*(\d+)\s*tỷ')
]

GREETINGS = {'hello', 'hi', 'chào', 'xin chào', 'hey'}
CONTRACTOR_PHRASES = [
    r"nhà thầu", r"thi công", r"xây dựng", r"công trình",
    r"tư vấn", r"báo giá", r"ngân sách", r"gợi ý"
]

PROJECT_TYPES = {
    'nhà phố':   [r"nhà phố", r"townhouse"],
    'biệt thự':  [r"biệt thự", r"villa"],
    'chung cư':  [r"chung cư", r"căn hộ", r"apartment"],
    'văn phòng': [r"văn phòng", r"office"],
    'nhà xưởng': [r"nhà xưởng", r"factory", r"xưởng"],
    'khách sạn': [r"khách sạn", r"hotel"],
    'nhà hàng':  [r"nhà hàng", r"restaurant"],
}

def _strip_accents(text: str) -> str:
    """Remove Vietnamese accents"""
    import unicodedata
    return ''.join(c for c in unicodedata.normalize('NFD', text) if unicodedata.category(c) != 'Mn')

def _contains_phrase(text: str, phrase: str) -> bool:
    """Check if text contains phrase"""
    return phrase.lower() in text.lower()

def detect_intent(msg: str) -> Tuple[ChatIntent, dict]:
    """Detect chat intent and extract information"""
    raw = msg or ""
    msg_lower = raw.lower().strip()
    msg_nod = _strip_accents(msg_lower)

    # 1) Greeting detection
    if len(msg_lower.split()) <= 4 and any(g in msg_lower for g in GREETINGS):
        return (ChatIntent.GREETING, {})

    # 2) Contractor intent detection
    has_contractor_intent = any(
        _contains_phrase(msg_lower, p) or _contains_phrase(msg_nod, _strip_accents(p))
        for p in CONTRACTOR_PHRASES
    )
    if not has_contractor_intent:
        return (ChatIntent.GENERAL, {})

    # 3) Extract information
    info = {}

    # Budget extraction
    for pat in BUDGET_PATTERNS:
        m = pat.search(msg_lower)
        if m:
            info['budget'] = m.group(0)
            break

    # Project type extraction
    for ptype, kws in PROJECT_TYPES.items():
        if any(_contains_phrase(msg_lower, k) or _contains_phrase(msg_nod, _strip_accents(k))
               for k in kws):
            info['project_type'] = ptype
            break

    # Location extraction
    for loc in [r"đà nẵng", r"da nang", r"hà nội", r"hanoi",
                r"hồ chí minh", r"ho chi minh", r"sài gòn", r"saigon"]:
        if _contains_phrase(msg_lower, loc) or _contains_phrase(msg_nod, _strip_accents(loc)):
            info['location'] = loc
            break

    # 4) Determine final intent
    if 'budget' in info and 'project_type' in info:
        return (ChatIntent.CONTRACTOR_FULL, info)
    elif 'budget' in info or 'project_type' in info or 'location' in info:
        return (ChatIntent.CONTRACTOR_PARTIAL, info)
    else:
        # If user asks about contractors but without specific info, still treat as contractor query
        # This handles queries like "cho tôi tất cả nhà thầu", "danh sách nhà thầu"
        return (ChatIntent.CONTRACTOR_PARTIAL, info)

def extract_contractor_info(chunks: List[dict], limit: int = 5, filter_criteria: dict = None) -> List[ContractorAction]:
    """Extract contractor information from chunks with optional filtering"""
    contractors = []
    seen = set()

    logger.info(f"Extracting contractors from {len(chunks)} chunks, limit={limit}, filter={filter_criteria}")

    for i, chunk in enumerate(chunks):
        if len(contractors) >= limit:
            break

        content = chunk.get('content', '')
        if '|' not in content:
            logger.debug(f"Chunk {i}: No pipe separator found")
            continue

        parts = [p.strip() for p in content.split('|') if p.strip()]
        if len(parts) < 9:
            logger.debug(f"Chunk {i}: Only {len(parts)} parts, need at least 9")
            continue

        contractor_id = parts[1]
        if contractor_id in seen:
            logger.debug(f"Chunk {i}: Contractor {contractor_id} already seen, skipping")
            continue

        # Extract data first
        try:
            # Extract specialties from field (parts[5]) if available
            specialties = []
            if len(parts) > 5 and parts[5]:
                # If field contains multiple specialties separated by comma
                specialties = [s.strip() for s in parts[5].split(',') if s.strip()]
                # If single value, add it as specialty
                if not specialties and parts[5].strip():
                    specialties = [parts[5].strip()]

            contractor = ContractorAction(
                contractor_id=contractor_id,
                contractor_name=parts[2],
                contractor_slug=parts[3],
                description=parts[4],
                budget_range=parts[6],
                rating=float(parts[8]) if parts[8].replace('.','').replace('-','').isdigit() else 4.0,
                specialties=specialties,
                location=parts[7],
                profile_url=f"{settings.FRONTEND_URL}/contractors/{contractor_id}",
                contact_url=f"{settings.FRONTEND_URL}/contractors/{contractor_id}?action=contact"
            )

            # Apply filters if provided
            if filter_criteria:
                # Filter by budget
                if 'budget' in filter_criteria:
                    budget_str = filter_criteria['budget']
                    # Extract number from budget string (e.g., "2 tỷ" -> 2)
                    import re
                    budget_match = re.search(r'(\d+)', budget_str)
                    if budget_match:
                        user_budget = float(budget_match.group(1))
                        # Parse contractor budget range (e.g., "2 tỷ - 8 tỷ")
                        contractor_budget = parts[6]
                        budget_parts = re.findall(r'(\d+\.?\d*)', contractor_budget)
                        if len(budget_parts) >= 2:
                            min_budget = float(budget_parts[0])
                            max_budget = float(budget_parts[1])
                            # Check if user budget is within range
                            if not (min_budget <= user_budget <= max_budget):
                                logger.debug(f"Chunk {i}: Budget {user_budget} tỷ not in range {min_budget}-{max_budget} tỷ")
                                continue

                # Filter by project type
                if 'project_type' in filter_criteria:
                    project_type = filter_criteria['project_type'].lower()
                    description_lower = parts[4].lower()
                    # Check if project type is mentioned in description
                    if project_type not in description_lower:
                        logger.debug(f"Chunk {i}: Project type '{project_type}' not found in description")
                        continue

            seen.add(contractor_id)
            contractors.append(contractor)
            logger.info(f"Chunk {i}: Successfully extracted contractor {parts[2]}")

        except (IndexError, ValueError) as e:
            logger.warning(f"Chunk {i}: Error parsing contractor {contractor_id}: {e}")
            continue

    logger.info(f"Extracted {len(contractors)} unique contractors from {len(chunks)} chunks (after filtering)")
    return contractors

def generate_greeting() -> str:
    """Generate greeting response"""
    return "Xin chào! Tôi có thể tư vấn nhà thầu xây dựng cho bạn. Bạn cần loại công trình nào và ngân sách bao nhiêu?"

def generate_missing_info(info: dict) -> str:
    """Generate response asking for missing information"""
    missing = []
    if 'budget' not in info:
        missing.append("ngân sách")
    if 'project_type' not in info:
        missing.append("loại công trình")
    
    return f"Để gợi ý chính xác, cho tôi biết thêm {' và '.join(missing)} nhé!\n\nVí dụ: 'Tìm nhà thầu xây nhà phố 3 tỷ'"

def generate_no_contractors(info: dict) -> str:
    """Generate response when no contractors found"""
    return f"""Chưa tìm thấy nhà thầu phù hợp với:
• Loại: {info.get('project_type', '?')}
• Ngân sách: {info.get('budget', '?')}

Thử điều chỉnh điều kiện hoặc liên hệ hotline nhé!"""
