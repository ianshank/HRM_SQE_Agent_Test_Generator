"""
Context builder for RAG-enhanced test generation.

Builds enriched context from multiple sources.
"""

from typing import Dict, Any, List, Optional
import logging

logger = logging.getLogger(__name__)


class ContextBuilder:
    """
    Build enriched context from RAG and requirements.
    
    Combines:
    - Historical test case examples from RAG
    - Technical stack information
    - Architecture patterns
    - Domain-specific knowledge
    """
    
    def __init__(self, rag_retriever: Optional[Any] = None):
        """
        Initialize context builder.
        
        Args:
            rag_retriever: Optional RAG retriever
        """
        self.rag_retriever = rag_retriever
        logger.info(f"ContextBuilder initialized (RAG: {rag_retriever is not None})")
    
    def build_generation_context(
        self,
        requirements: Dict[str, Any],
        include_tech_stack: bool = True,
        include_architecture: bool = True,
        include_similar_tests: bool = True,
        max_similar_tests: int = 5,
    ) -> str:
        """
        Build comprehensive context for test generation.
        
        Args:
            requirements: Requirements dictionary (Epic)
            include_tech_stack: Include tech stack in context
            include_architecture: Include architecture in context
            include_similar_tests: Include similar tests from RAG
            max_similar_tests: Max number of similar tests
            
        Returns:
            Context string
        """
        logger.debug("Building generation context")
        
        context_parts = []
        
        context_parts.append("# Test Generation Context")
        context_parts.append("")
        
        if include_tech_stack:
            tech_context = self._build_tech_stack_context(requirements)
            if tech_context:
                context_parts.append(tech_context)
                context_parts.append("")
        
        if include_architecture:
            arch_context = self._build_architecture_context(requirements)
            if arch_context:
                context_parts.append(arch_context)
                context_parts.append("")
        
        if include_similar_tests and self.rag_retriever:
            similar_context = self._build_similar_tests_context(
                requirements,
                max_similar_tests
            )
            if similar_context:
                context_parts.append(similar_context)
                context_parts.append("")
        
        acceptance_context = self._build_acceptance_criteria_context(requirements)
        if acceptance_context:
            context_parts.append(acceptance_context)
            context_parts.append("")
        
        context_string = "\n".join(context_parts)
        
        logger.debug(f"Context built: {len(context_string)} characters")
        
        return context_string
    
    def _build_tech_stack_context(self, requirements: Dict[str, Any]) -> str:
        """Build tech stack context."""
        tech_stack = requirements.get("tech_stack", [])
        
        if not tech_stack:
            for story in requirements.get("user_stories", []):
                tech_stack.extend(story.get("tech_stack", []))
        
        if not tech_stack:
            return ""
        
        tech_stack = list(set(tech_stack))
        
        parts = [
            "## Technology Stack",
            "",
            f"The system uses the following technologies: {', '.join(tech_stack)}",
            "",
            "Test cases should consider:",
        ]
        
        if "FastAPI" in tech_stack or "Python" in tech_stack:
            parts.append("- Python/FastAPI patterns (pytest, httpx)")
        
        if "React" in tech_stack or "TypeScript" in tech_stack:
            parts.append("- React/TypeScript patterns (Jest, RTL)")
        
        if any(db in tech_stack for db in ["PostgreSQL", "MySQL", "MongoDB"]):
            parts.append("- Database integration and transactions")
        
        if any(q in tech_stack for q in ["RabbitMQ", "Kafka", "Redis"]):
            parts.append("- Message queue and async processing")
        
        return "\n".join(parts)
    
    def _build_architecture_context(self, requirements: Dict[str, Any]) -> str:
        """Build architecture context."""
        architecture = requirements.get("architecture", "")
        
        if not architecture:
            return ""
        
        parts = [
            "## Architecture",
            "",
            f"Architecture Pattern: {architecture}",
            "",
        ]
        
        if "microservice" in architecture.lower():
            parts.extend([
                "Test considerations:",
                "- Service isolation and boundaries",
                "- Inter-service communication",
                "- API contract testing",
                "- Service mesh integration",
            ])
        elif "monolith" in architecture.lower():
            parts.extend([
                "Test considerations:",
                "- Module integration",
                "- End-to-end workflows",
                "- Database transactions",
            ])
        
        return "\n".join(parts)
    
    def _build_similar_tests_context(
        self,
        requirements: Dict[str, Any],
        max_tests: int,
    ) -> str:
        """Build context from similar historical tests."""
        if not self.rag_retriever:
            return ""
        
        try:
            similar_tests = self.rag_retriever.retrieve_similar_test_cases(
                requirements,
                top_k=max_tests,
                min_similarity=0.7
            )
            
            if not similar_tests:
                return ""
            
            context = self.rag_retriever.build_context(
                requirements,
                similar_tests,
                include_metadata=True
            )
            
            return context
            
        except Exception as e:
            logger.warning(f"Failed to retrieve similar tests: {e}")
            return ""
    
    def _build_acceptance_criteria_context(
        self,
        requirements: Dict[str, Any]
    ) -> str:
        """Build context from acceptance criteria."""
        parts = ["## Acceptance Criteria to Cover"]
        parts.append("")
        
        story_count = 0
        criteria_count = 0
        
        for story in requirements.get("user_stories", []):
            criteria = story.get("acceptance_criteria", [])
            if criteria:
                story_count += 1
                parts.append(f"**{story.get('summary', 'Story')}:**")
                
                for criterion in criteria:
                    criterion_text = (
                        criterion.get("criteria") 
                        if isinstance(criterion, dict) 
                        else str(criterion)
                    )
                    parts.append(f"- {criterion_text}")
                    criteria_count += 1
                
                parts.append("")
        
        if criteria_count == 0:
            return ""
        
        parts.insert(
            2,
            f"Total: {story_count} user stories with {criteria_count} acceptance criteria"
        )
        parts.insert(3, "")
        
        return "\n".join(parts)
    
    def build_compact_context(self, requirements: Dict[str, Any]) -> str:
        """Build compact context for token-limited scenarios."""
        parts = []
        
        tech_stack = requirements.get("tech_stack", [])
        if tech_stack:
            parts.append(f"Tech: {', '.join(tech_stack[:3])}")
        
        architecture = requirements.get("architecture", "")
        if architecture:
            parts.append(f"Arch: {architecture}")
        
        story_count = len(requirements.get("user_stories", []))
        parts.append(f"{story_count} stories")
        
        return " | ".join(parts)
