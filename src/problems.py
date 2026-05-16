"""
Data models and loader for coding problems and test cases.

This module combines the problem models and loading functionality
to access LiveCodeBench dataset directly via HuggingFace.
"""

import json
import zlib
import pickle
import base64
from enum import Enum
from datetime import datetime
from dataclasses import dataclass
from typing import List, Optional
from datasets import load_dataset


class Platform(Enum):
    LEETCODE = "leetcode"
    CODEFORCES = "codeforces"
    ATCODER = "atcoder"


class Difficulty(Enum):
    EASY = "easy"
    MEDIUM = "medium"
    HARD = "hard"
    # AtCoder-specific fine-grained difficulties
    ATCODER_EASY = "atcoder_easy"       # ABC A, B problems
    ATCODER_MEDIUM = "atcoder_medium"   # ABC C, D problems  
    ATCODER_HARD = "atcoder_hard"       # ABC E, F problems
    ATCODER_EXPERT = "atcoder_expert"   # ARC A, B, AGC A problems
    ATCODER_MASTER = "atcoder_master"   # ARC C, D, AGC B problems
    ATCODER_GRANDMASTER = "atcoder_grandmaster"  # ARC E, F, AGC C+ problems


class TestType(Enum):
    STDIN = "stdin"
    FUNCTIONAL = "functional"


@dataclass
class Test:
    input: str
    output: str
    testtype: TestType

    def __post_init__(self):
        self.testtype = TestType(self.testtype)


@dataclass
class CodeGenerationProblem:
    question_title: str
    question_content: str
    platform: Platform
    question_id: str
    contest_id: str
    contest_date: datetime
    starter_code: str
    difficulty: Difficulty
    public_test_cases: list[Test]
    private_test_cases: list[Test]
    metadata: dict

    def __post_init__(self):
        self.platform = Platform(self.platform)
        self.difficulty = Difficulty(self.difficulty)
        self.contest_date = datetime.fromisoformat(self.contest_date)

        self.public_test_cases = json.loads(self.public_test_cases)  # type: ignore
        self.public_test_cases = [Test(**t) for t in self.public_test_cases]

        try:
            self.private_test_cases = json.loads(self.private_test_cases)  # type: ignore
        except:
            self.private_test_cases = json.loads(
                pickle.loads(
                    zlib.decompress(
                        base64.b64decode(self.private_test_cases.encode("utf-8"))  # type: ignore
                    )
                )
            )  # type: ignore
        self.private_test_cases = [Test(**t) for t in self.private_test_cases]

        self.metadata = json.loads(self.metadata)  # type: ignore

    def insert_output(self, output_list: list[str], code_list: list[str]) -> dict:
        return {
            "question_title": self.question_title,
            "question_content": self.question_content,
            "platform": self.platform.value,
            "question_id": self.question_id,
            "contest_id": self.contest_id,
            "contest_date": self.contest_date.isoformat(),
            "starter_code": self.starter_code,
            "difficulty": self.difficulty.value,
            "output_list": output_list,
            "code_list": code_list,
        }

    def insert_output_evaluation(
        self,
        output_list: list[str],
        code_list: list[str],
        graded_list: list[bool],
        **kwargs,
    ) -> dict:
        output = self.insert_output(output_list, code_list)
        output["graded_list"] = graded_list
        output["pass@1"] = graded_list.count(True) / len(graded_list)
        for k, v in kwargs.items():
            output[k] = v
        return output

    def get_evaluation_sample(self):
        return {
            "input_output": json.dumps(
                {
                    "inputs": [
                        t.input
                        for t in self.public_test_cases + self.private_test_cases
                    ],
                    "outputs": [
                        t.output
                        for t in self.public_test_cases + self.private_test_cases
                    ],
                    "fn_name": self.metadata.get("func_name", None),
                }
            ),
        }


def get_atcoder_granular_difficulty(problem: 'CodeGenerationProblem') -> Difficulty:
    """
    Map AtCoder problems to granular difficulty based on contest type and problem position.
    
    Based on typical AtCoder difficulty progression:
    - ABC (AtCoder Beginner Contest): A/B (easy), C/D (medium), E/F (hard)
    - ARC (AtCoder Regular Contest): A/B (expert), C/D (master), E/F (grandmaster)
    - AGC (AtCoder Grand Contest): A (expert), B (master), C+ (grandmaster)
    """
    if problem.platform != Platform.ATCODER:
        return problem.difficulty
    
    contest_id = problem.contest_id.lower()
    question_id = problem.question_id.lower()
    
    # Extract problem letter (last character of question_id usually)
    problem_letter = question_id[-1] if question_id else 'a'
    
    # ABC contests (Beginner)
    if 'abc' in contest_id:
        if problem_letter in ['a', 'b']:
            return Difficulty.ATCODER_EASY
        elif problem_letter in ['c', 'd']:
            return Difficulty.ATCODER_MEDIUM
        else:  # e, f, g, h
            return Difficulty.ATCODER_HARD
    
    # ARC contests (Regular)
    elif 'arc' in contest_id:
        if problem_letter in ['a', 'b']:
            return Difficulty.ATCODER_EXPERT
        elif problem_letter in ['c', 'd']:
            return Difficulty.ATCODER_MASTER
        else:  # e, f
            return Difficulty.ATCODER_GRANDMASTER
    
    # AGC contests (Grand)
    elif 'agc' in contest_id:
        if problem_letter == 'a':
            return Difficulty.ATCODER_EXPERT
        elif problem_letter == 'b':
            return Difficulty.ATCODER_MASTER
        else:  # c, d, e, f
            return Difficulty.ATCODER_GRANDMASTER
    
    # Other contests - fall back to original difficulty
    else:
        return problem.difficulty


def load_code_generation_dataset(
    release_version: str = "v6", 
    start_date: Optional[str] = None, 
    end_date: Optional[str] = None, 
    difficulty: Optional[str] = None
) -> List[CodeGenerationProblem]:
    """
    Load problems from LiveCodeBench dataset via HuggingFace.
    
    Args:
        release_version: Dataset version (release_v1, release_v6, etc.)
        start_date: Filter problems after this date (YYYY-MM-DD)
        end_date: Filter problems before this date (YYYY-MM-DD) 
        difficulty: Filter by difficulty (easy, medium, hard)
        
    Returns:
        List of CodeGenerationProblem objects
    """
    dataset = load_dataset(
        "livecodebench/code_generation_lite", 
        split="test", 
        version_tag=release_version, 
        trust_remote_code=True
    )
    dataset = [CodeGenerationProblem(**p) for p in dataset]  # type: ignore
    
    if start_date is not None:
        p_start_date = datetime.strptime(start_date, "%Y-%m-%d")
        dataset = [e for e in dataset if p_start_date <= e.contest_date]

    if end_date is not None:
        p_end_date = datetime.strptime(end_date, "%Y-%m-%d")
        dataset = [e for e in dataset if e.contest_date <= p_end_date]

    if difficulty is not None:
        if isinstance(difficulty, str):
            difficulty = Difficulty(difficulty)
        
        # For AtCoder-specific difficulties, use granular mapping
        if difficulty.value.startswith('atcoder_'):
            dataset = [e for e in dataset if get_atcoder_granular_difficulty(e) == difficulty]
        else:
            dataset = [e for e in dataset if e.difficulty == difficulty]

    print(f"Loaded {len(dataset)} problems")
    return dataset


def load_code_generation_dataset_full(release_version: str = "release_v6") -> List[CodeGenerationProblem]:
    """
    Load full dataset (not lite version).
    
    Args:
        release_version: Dataset version
        
    Returns:
        List of CodeGenerationProblem objects
    """
    dataset = load_dataset("livecodebench/code_generation", split="test")
    dataset = [CodeGenerationProblem(**p) for p in dataset]  # type: ignore
    print(f"Loaded {len(dataset)} problems")
    return dataset