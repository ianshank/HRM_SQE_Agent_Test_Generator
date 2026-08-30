"""
Natural Language Requirements Parser.

Parses plain text requirements into structured Epic/UserStory format.
Supports multiple formats: free text, Gherkin, numbered lists.
"""

import re
import logging
from typing import List, Dict, Tuple, Optional
from pathlib import Path

from .schemas import Epic, UserStory, AcceptanceCriteria

logger = logging.getLogger(__name__)


class NaturalLanguageParser:
    """
    Parse natural language requirements into structured format.
    
    Supports multiple input patterns:
    - Gherkin-style (Given/When/Then)
    - User story format (As a... I want... So that...)
    - Numbered requirements
    - Free text with headers
    """
    
    # Section terminators for acceptance criteria parsing
    SECTION_TERMINATORS = ('User Story', 'Related Task', 'Epic', 'US', 'Story')
    
    # Regex patterns for detection
    EPIC_PATTERNS = [
        r'^Epic:?\s*(.+)$',
        r'^#\s+(.+)$',  # Markdown header
        r'^\*\*Epic\*\*:?\s*(.+)$',  # Bold Epic
    ]
    
    USER_STORY_PATTERNS = [
        r'(?:User Story|US|Story):?\s*(\d*):?\s*(.+)',
        r'As an?\s+(.+?),?\s+I want\s+(.+?)(?:,?\s+so that\s+(.+))?',
        r'^\d+\.\s+(.+)',  # Numbered item
    ]
    
    AC_PATTERNS = [
        r'(?:AC|Acceptance Criteria?):?\s*(.+)',
        r'Given\s+(.+?)\s+When\s+(.+?)\s+Then\s+(.+)',
        r'^[-*]\s+(.+)',  # Bullet points
    ]
    
    def __init__(self):
        """Initialize parser with compiled regex patterns."""
        self.epic_re = [re.compile(p, re.IGNORECASE | re.MULTILINE) for p in self.EPIC_PATTERNS]
        self.us_re = [re.compile(p, re.IGNORECASE | re.MULTILINE) for p in self.USER_STORY_PATTERNS]
        self.ac_re = [re.compile(p, re.IGNORECASE | re.MULTILINE) for p in self.AC_PATTERNS]
        
        logger.info("NaturalLanguageParser initialized with pattern matching")
    
    def parse(self, text: str, filename: Optional[str] = None) -> Epic:
        """
        Parse natural language text into structured Epic.
        
        Args:
            text: Raw requirements text
            filename: Optional filename for epic ID generation
            
        Returns:
            Epic object with parsed structure
            
        Raises:
            ValueError: If text is empty or cannot be parsed
        """
        if not text or not text.strip():
            raise ValueError("Empty text provided for parsing")
        
        logger.info(f"Parsing requirements text ({len(text)} chars)")
        
        # Extract components
        epic_title = self._extract_epic_title(text, filename)
        user_stories = self._extract_user_stories(text)
        
        # If no user stories found, treat entire text as single story
        if not user_stories:
            logger.warning("No user stories detected, creating single story from text")
            user_stories = [self._create_fallback_story(text)]
        
        epic = Epic(
            epic_id=self._generate_epic_id(epic_title, filename),
            title=epic_title,
            user_stories=user_stories
        )
        
        logger.info(f"Parsed epic '{epic.title}' with {len(epic.user_stories)} user stories")
        return epic
    
    def _extract_epic_title(self, text: str, filename: Optional[str] = None) -> str:
        """Extract epic title from text."""
        lines = text.split('\n')
        
        # Try patterns
        for pattern in self.epic_re:
            for line in lines[:5]:  # Check first 5 lines
                match = pattern.match(line.strip())
                if match:
                    title = match.group(1).strip()
                    logger.debug(f"Found epic title via pattern: {title}")
                    return title
        
        # Fallback: use first non-empty line
        for line in lines[:3]:
            line = line.strip()
            if line and len(line) > 5:
                logger.debug(f"Using first line as epic title: {line[:50]}...")
                return line
        
        # Last resort: use filename or generic title
        if filename:
            title = Path(filename).stem.replace('_', ' ').title()
            logger.debug(f"Using filename as epic title: {title}")
            return title
        
        logger.debug("Using generic epic title")
        return "Requirements"
    
    def _extract_user_stories(self, text: str) -> List[UserStory]:
        """Extract user stories from text."""
        stories = []
        lines = text.split('\n')
        
        # Try to detect story boundaries
        story_sections = self._split_into_sections(lines)
        
        for idx, section in enumerate(story_sections, 1):
            story = self._parse_story_section(section, idx)
            if story:
                stories.append(story)
        
        logger.debug(f"Extracted {len(stories)} user stories")
        return stories
    
    def _split_into_sections(self, lines: List[str]) -> List[List[str]]:
        """Split text into logical sections for user stories."""
        sections = []
        current_section = []
        
        for line in lines:
            line = line.strip()
            
            # Check if line starts a new section
            is_new_section = any([
                line.startswith(('US', 'User Story', 'Story')),
                re.match(r'^\d+\.', line),  # Numbered item
                (line.startswith('#') and len(line) > 1),  # Header
                (line.startswith('As a') or line.startswith('As an')),  # User story format
            ])
            
            if is_new_section and current_section:
                # Save previous section
                sections.append(current_section)
                current_section = [line]
            elif line:
                current_section.append(line)
        
        # Add last section
        if current_section:
            sections.append(current_section)
        
        return sections if sections else [lines]
    
    def _parse_story_section(self, lines: List[str], story_num: int) -> Optional[UserStory]:
        """Parse a section into a UserStory."""
        if not lines:
            return None
        
        # Join lines for analysis
        text = ' '.join(lines)
        
        # Extract summary and description
        summary, description = self._extract_story_content(lines)
        
        if not summary:
            logger.debug(f"Skipping section {story_num}: no valid summary")
            return None
        
        # Extract acceptance criteria
        acceptance_criteria = self._extract_acceptance_criteria(lines)
        
        story = UserStory(
            id=f"US-{story_num:03d}",
            summary=summary,
            description=description,
            acceptance_criteria=acceptance_criteria
        )
        
        logger.debug(f"Parsed story: {story.summary[:50]}...")
        return story
    
    def _extract_story_content(self, lines: List[str]) -> Tuple[str, str]:
        """Extract summary and description from lines."""
        # Try user story pattern first
        text = ' '.join(lines)
        
        for pattern in self.us_re:
            match = pattern.search(text)
            if match:
                groups = match.groups()
                if len(groups) >= 2:
                    # "As a X, I want Y" format
                    summary = f"{groups[0]} - {groups[1]}"
                    description = groups[2] if len(groups) > 2 and groups[2] else text
                    return summary[:200], description[:500]
        
        # Fallback: use first line as summary, rest as description
        if lines:
            summary = lines[0].strip().lstrip('#*- ').strip()
            description = ' '.join(lines[1:]).strip() if len(lines) > 1 else summary
            return summary[:200], description[:500]
        
        return "", ""
    
    def _extract_acceptance_criteria(self, lines: List[str]) -> List[AcceptanceCriteria]:
        """Extract acceptance criteria from lines."""
        criteria = []
        in_criteria_section = False
        
        for line in lines:
            line = line.strip()
            
            # Detect start of acceptance criteria section
            if re.match(r'^Acceptance\s+Criteria:?\s*$', line, re.IGNORECASE):
                in_criteria_section = True
                continue
            
            # If in criteria section, treat each non-empty line as a criterion
            if in_criteria_section:
                if not line or line.startswith(self.SECTION_TERMINATORS):
                    in_criteria_section = False
                    continue
                if len(line) > 5 and not line.startswith('#'):
                    criteria.append(AcceptanceCriteria(criteria=line))
                    continue
            
            # Check AC patterns (bullet points, numbered lists)
            for pattern in self.ac_re:
                match = pattern.match(line)
                if match:
                    # Extract criterion text
                    if 'Given' in line and 'When' in line and 'Then' in line:
                        # Gherkin format
                        criterion_text = line
                    else:
                        criterion_text = match.group(1).strip()
                    
                    if criterion_text and len(criterion_text) > 5:
                        criteria.append(AcceptanceCriteria(criteria=criterion_text))
                    break
        
        logger.debug(f"Extracted {len(criteria)} acceptance criteria")
        return criteria
    
    def _create_fallback_story(self, text: str) -> UserStory:
        """Create a single user story from entire text."""
        lines = [l.strip() for l in text.split('\n') if l.strip()]
        
        summary = lines[0][:200] if lines else "Requirement"
        description = ' '.join(lines[:10])[:500]  # First 10 lines
        
        # Extract any bullet points as acceptance criteria
        ac = []
        for line in lines:
            if line.startswith(('-', '*', '•')):
                criterion = line.lstrip('-*• ').strip()
                if len(criterion) > 5:
                    ac.append(AcceptanceCriteria(criteria=criterion))
        
        logger.debug(f"Created fallback story with {len(ac)} criteria")
        
        return UserStory(
            id="US-001",
            summary=summary,
            description=description,
            acceptance_criteria=ac if ac else []
        )
    
    def _generate_epic_id(self, title: str, filename: Optional[str] = None) -> str:
        """Generate epic ID from title or filename."""
        if filename:
            base = Path(filename).stem
        else:
            base = re.sub(r'[^a-zA-Z0-9]+', '_', title)[:20]
        
        epic_id = f"EPIC-{base.upper()}"
        return epic_id


def parse_natural_language_requirements(
    text: str,
    filename: Optional[str] = None
) -> Epic:
    """
    Convenience function to parse requirements.
    
    Args:
        text: Natural language requirements text
        filename: Optional filename for ID generation
        
    Returns:
        Parsed Epic object
    """
    parser = NaturalLanguageParser()
    return parser.parse(text, filename)
