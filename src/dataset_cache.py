"""
Dataset caching system to avoid repeated loading of LiveCodeBench dataset.

This module provides caching functionality to store and retrieve dataset
problems efficiently, supporting both full dataset caching and individual
problem lookup.
"""

import json
import pickle
import logging
from pathlib import Path
from typing import Dict, List, Optional, Set

from problems import load_code_generation_dataset, CodeGenerationProblem

logger = logging.getLogger(__name__)


class DatasetCache:
    """Manages caching of LiveCodeBench dataset problems."""
    
    def __init__(self, cache_dir: str = ".cache"):
        """Initialize dataset cache.
        
        Args:
            cache_dir: Directory to store cache files
        """
        self.cache_dir = Path(cache_dir)
        self.cache_dir.mkdir(exist_ok=True)
        self._memory_cache: Dict[str, List[CodeGenerationProblem]] = {}
        
    def _get_cache_file(self, release_version: str) -> Path:
        """Get cache file path for a release version."""
        return self.cache_dir / f"dataset_{release_version}.pkl"
    
    def _get_index_file(self, release_version: str) -> Path:
        """Get index file path for a release version."""
        return self.cache_dir / f"index_{release_version}.json"
    
    def _load_from_cache(self, release_version: str) -> Optional[List[CodeGenerationProblem]]:
        """Load problems from cache file."""
        cache_file = self._get_cache_file(release_version)
        
        if not cache_file.exists():
            logger.info(f"Cache for {release_version} is missing")
            return None
        
        try:
            with open(cache_file, 'rb') as f:
                problems = pickle.load(f)
            print(f"Using cached dataset {release_version} ({len(problems)} problems)")
            logger.debug(f"Loaded {len(problems)} problems from cache: {release_version}")
            return problems
        except Exception as e:
            logger.warning(f"Failed to load cache: {e}")
            return None
    
    def _save_to_cache(self, release_version: str, problems: List[CodeGenerationProblem]):
        """Save problems to cache file."""
        cache_file = self._get_cache_file(release_version)
        index_file = self._get_index_file(release_version)
        
        try:
            # Save full dataset
            with open(cache_file, 'wb') as f:
                pickle.dump(problems, f)
            
            # Save index for quick lookups
            index = {
                problem.question_id: {
                    'title': problem.question_title,
                    'platform': problem.platform.value,
                    'difficulty': problem.difficulty.value,
                    'contest_date': problem.contest_date.isoformat()
                }
                for problem in problems
            }
            
            with open(index_file, 'w') as f:
                json.dump(index, f, indent=2)
            
            logger.info(f"Cached {len(problems)} problems for {release_version}")
            
        except Exception as e:
            logger.error(f"Failed to save cache: {e}")
    
    def get_problems(self, release_version: str = "v6", 
                    force_refresh: bool = False) -> List[CodeGenerationProblem]:
        """Get problems for a release version, using cache when possible.
        
        Args:
            release_version: Dataset version to load
            force_refresh: Force reload from dataset even if cached
            
        Returns:
            List of CodeGenerationProblem objects
        """
        # Check memory cache first
        if not force_refresh and release_version in self._memory_cache:
            logger.debug(f"Using memory cache for {release_version}")
            return self._memory_cache[release_version]
        
        # Try to load from disk cache
        if not force_refresh:
            cached_problems = self._load_from_cache(release_version)
            if cached_problems is not None:
                self._memory_cache[release_version] = cached_problems
                return cached_problems
        
        # Load from dataset and cache
        print(f"Loading {release_version} from dataset (first time - will be cached)...")
        logger.info(f"Loading {release_version} from dataset...")
        try:
            problems = load_code_generation_dataset(release_version=release_version)
            
            # Cache the results
            self._save_to_cache(release_version, problems)
            self._memory_cache[release_version] = problems
            
            return problems
            
        except Exception as e:
            logger.error(f"Failed to load dataset: {e}")
            raise
    
    def find_problem(self, problem_id: str, 
                    release_version: str = "v6") -> Optional[CodeGenerationProblem]:
        """Find a specific problem by ID.
        
        Args:
            problem_id: Problem ID to find
            release_version: Dataset version to search in
            
        Returns:
            CodeGenerationProblem if found, None otherwise
        """
        # Try to use index for quick lookup first
        index_file = self._get_index_file(release_version)
        if index_file.exists():
            try:
                with open(index_file, 'r') as f:
                    index = json.load(f)
                
                if problem_id not in index:
                    logger.info(f"Problem {problem_id} not found in {release_version} index")
                    return None
                
                logger.debug(f"Found {problem_id} in index, loading from dataset")
                    
            except Exception as e:
                logger.warning(f"Failed to read index: {e}")
        
        # Load full dataset and search
        logger.debug(f"Searching for {problem_id} in full dataset")
        problems = self.get_problems(release_version)
        for problem in problems:
            if problem.question_id == problem_id:
                logger.debug(f"Found problem {problem_id}")
                return problem
        
        logger.info(f"Problem {problem_id} not found in {release_version}")
        return None
    
    def get_available_problems(self, release_version: str = "v6") -> Set[str]:
        """Get set of available problem IDs for a release version.
        
        Args:
            release_version: Dataset version to check
            
        Returns:
            Set of problem IDs
        """
        # Try index file first for efficiency
        index_file = self._get_index_file(release_version)
        if index_file.exists():
            try:
                with open(index_file, 'r') as f:
                    index = json.load(f)
                return set(index.keys())
            except Exception as e:
                logger.warning(f"Failed to read index: {e}")
        
        # Fallback to loading full dataset
        problems = self.get_problems(release_version)
        return {problem.question_id for problem in problems}
    
    def list_versions(self) -> List[str]:
        """List available cached versions."""
        versions = []
        for cache_file in self.cache_dir.glob("dataset_*.pkl"):
            version = cache_file.stem.replace("dataset_", "")
            versions.append(version)
        return sorted(versions)
    
    def clear_cache(self, release_version: Optional[str] = None):
        """Clear cache files.
        
        Args:
            release_version: Specific version to clear, or None for all
        """
        if release_version:
            # Clear specific version
            cache_file = self._get_cache_file(release_version)
            index_file = self._get_index_file(release_version)
            
            for file in [cache_file, index_file]:
                if file.exists():
                    file.unlink()
                    logger.info(f"Removed cache file: {file}")
            
            # Clear from memory cache
            self._memory_cache.pop(release_version, None)
        else:
            # Clear all cache files
            for file in self.cache_dir.glob("dataset_*.pkl"):
                file.unlink()
            for file in self.cache_dir.glob("index_*.json"):
                file.unlink()
            self._memory_cache.clear()
            logger.info("Cleared all cache files")


# Global cache instance
_cache = DatasetCache()


def get_cached_problems(release_version: str = "v6", 
                       force_refresh: bool = False) -> List[CodeGenerationProblem]:
    """Get problems using the global cache instance."""
    return _cache.get_problems(release_version, force_refresh)


def find_cached_problem(problem_id: str, 
                       release_version: str = "v6") -> Optional[CodeGenerationProblem]:
    """Find a problem using the global cache instance."""
    return _cache.find_problem(problem_id, release_version)


def get_available_problems(release_version: str = "v6") -> Set[str]:
    """Get available problem IDs using the global cache instance."""
    return _cache.get_available_problems(release_version)


def clear_dataset_cache(release_version: Optional[str] = None):
    """Clear cache using the global cache instance."""
    _cache.clear_cache(release_version)