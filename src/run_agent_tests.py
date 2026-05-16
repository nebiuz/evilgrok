#!/usr/bin/env python3
"""
Main CLI for automated agent testing.

This script provides a command-line interface for running automated tests
of coding agents on LiveCodeBench problems.
"""

import argparse
import json
import sys
import yaml
import tempfile
from datetime import datetime
from pathlib import Path
from typing import List, Dict, Any

# Add src directory to path for imports
# This allows running from project root: python src/run_agent_tests.py
src_dir = Path(__file__).parent
if src_dir.name == 'src':
    # Running from root directory, add src to path
    sys.path.insert(0, str(src_dir))
else:
    # Running from src directory, add current directory
    sys.path.insert(0, str(src_dir))

from automated_tester import AutomatedTester
from env_utils import load_env_from_dotenv, build_subprocess_env
from dataset_cache import get_cached_problems, get_available_problems


def generate_descriptive_run_name(agents: List[Dict], problems: List[str], release_version: str, 
                                  difficulty: str = None, timestamp_suffix: str = None) -> str:
    """Generate a descriptive run directory name based on test configuration."""
    # Agent names
    agent_names = sorted([agent['name'] for agent in agents])
    agents_str = '_'.join(agent_names)
    
    # Release version
    version_str = release_version.replace('release_', '').replace('_', '')
    
    # Problem count and difficulty
    problem_count = len(problems)
    if difficulty:
        problems_str = f"{problem_count}p_{difficulty}"
    else:
        # Try to infer difficulty from problem IDs or use generic count
        problems_str = f"{problem_count}p"
    
    # Build descriptive name
    parts = [agents_str, version_str, problems_str]
    
    # Add timestamp only if needed for disambiguation
    if timestamp_suffix:
        parts.append(timestamp_suffix)
    
    return '_'.join(parts)
import random


def load_config(config_path: str) -> Dict[str, Any]:
    """Load configuration from YAML file."""
    with open(config_path, 'r') as f:
        return yaml.safe_load(f)


def get_problems_from_config(config: Dict[str, Any]) -> List[str]:
    """Get list of problems to test based on configuration."""
    # If specific problems are listed, use those (no dataset loading needed)
    if config.get('specific_problems'):
        return config['specific_problems']
    
    # Otherwise use problem filters with caching
    filters = config.get('problem_filters', {})
    release_version = filters.get('release_version', 'v6')
    
    try:
        print(f"Loading dataset {release_version} (using cache if available)...")
        # Load problems using cached dataset loader
        problems = get_cached_problems(release_version=release_version)
        
        # Apply date filters
        if filters.get('date_range', {}).get('start'):
            start_date = datetime.strptime(filters['date_range']['start'], "%Y-%m-%d")
            problems = [p for p in problems if p.contest_date >= start_date]
            
        if filters.get('date_range', {}).get('end'):
            end_date = datetime.strptime(filters['date_range']['end'], "%Y-%m-%d")
            problems = [p for p in problems if p.contest_date <= end_date]
        
        # Filter by difficulty
        if 'difficulties' in filters:
            from problems import get_atcoder_granular_difficulty
            difficulty_values = filters['difficulties']
            filtered_problems = []
            for p in problems:
                # For AtCoder problems, check both original and granular difficulty
                if p.platform.value == 'atcoder':
                    granular_diff = get_atcoder_granular_difficulty(p)
                    if p.difficulty.value in difficulty_values or granular_diff.value in difficulty_values:
                        filtered_problems.append(p)
                else:
                    if p.difficulty.value in difficulty_values:
                        filtered_problems.append(p)
            problems = filtered_problems
        
        # Filter by platform
        if 'platforms' in filters:
            platform_values = filters['platforms']
            problems = [p for p in problems if p.platform.value in platform_values]
        
        # Limit number of problems
        max_problems = filters.get('max_problems', len(problems))
        problems = problems[:max_problems]
        
        print(f"Filtered to {len(problems)} problems from dataset")
        return [p.question_id for p in problems]
        
    except Exception as e:
        print(f"Error loading problems from dataset: {e}")
        # Return some fallback problems
        return ['leetcode_1', 'leetcode_2']


def _run_llm_detection(run_dir: Path, args):
    """Run LLM reward hacking detection if XAI_API_KEY is available."""
    import os
    if os.environ.get('XAI_API_KEY'):
        try:
            print("\n🔎 Running LLM reward hacking judge (Grok)...")
            import subprocess
            judge_script = Path('src') / 'llm_reward_hacking_detector.py'
            judge_out = run_dir / 'llm_reward_hacking_results.json'
            cmd = [
                sys.executable,
                str(judge_script),
                str(run_dir),
                '--output', str(judge_out)
            ]
            result = subprocess.run(
                cmd,
                text=True,
                capture_output=False,
                env=build_subprocess_env(required_vars=['XAI_API_KEY'])
            )
            print(f"LLM judge completed with exit code {result.returncode}")
            if judge_out.exists():
                print(f"LLM judge results saved to: {judge_out}")
        except Exception as e:
            print(f"⚠️  Failed to run LLM reward hacking judge: {e}")
    else:
        print("\nℹ️  Skipping LLM reward hacking judge (XAI_API_KEY not set)")


def main():
    parser = argparse.ArgumentParser(description='Reward hacking detector')
    
    # Configuration
    parser.add_argument('--config', '-c', 
                       default='test_config.yaml',
                       help='Configuration file path')
    
    # Agent selection
    parser.add_argument('--agent', '-a',
                       help='Specific agent to test (overrides config)')
    parser.add_argument('--agents',
                       nargs='+',
                       help='Multiple agents to test (overrides config)')
    
    # Model selection
    parser.add_argument('--model',
                       help='Model to use for Grok agent (e.g., grok-4.3, grok-4, grok-3, grok-3-mini)')
    parser.add_argument('--reasoning-effort',
                       choices=['low', 'medium', 'high'],
                       help='Reasoning effort for Grok reasoning models (low/medium/high)')
    
    # Problem selection
    parser.add_argument('--problem', '-p',
                       help='Specific problem ID to test')
    parser.add_argument('--problems',
                       nargs='+',
                       help='Multiple problem IDs to test')
    parser.add_argument('--random', '-r',
                       action='store_true',
                       help='Select a random problem to test')
    
    parser.add_argument('--difficulty', '-d',
                       choices=['easy', 'medium', 'hard', 
                               'atcoder_easy', 'atcoder_medium', 'atcoder_hard',
                               'atcoder_expert', 'atcoder_master', 'atcoder_grandmaster'],
                       help='Filter problems by difficulty (including AtCoder-specific levels)')
    
    parser.add_argument('--platform',
                       choices=['leetcode', 'atcoder', 'codeforces'],
                       help='Filter problems by platform')
    
    parser.add_argument('--max-problems', '-m',
                       type=int,
                       help='Maximum number of problems to test')
    
    parser.add_argument('--release-version',
                       default='v6',
                       help='Dataset release version (release_v1, release_v6, etc.)')
    
    parser.add_argument('--quick',
                       action='store_true', 
                       help='Quick mode: bypass config file, use minimal setup')
    
    # Test settings
    parser.add_argument('--timeout', '-t',
                       type=int,
                       help='Timeout per test in seconds')
    
    parser.add_argument('--no-cleanup',
                       action='store_true',
                       help='Keep workspace directories after testing')
    
    parser.add_argument('--scenario', '-s',
                       help='Use predefined test scenario from config')
    
    # Parallel execution
    parser.add_argument('--parallel',
                       action='store_true',
                       help='Run agents in parallel')
    parser.add_argument('--max-workers',
                       type=int,
                       help='Maximum number of parallel workers (default: 4)')
    
    # Output settings
    parser.add_argument('--output', '-o',
                       help='Output file for results')
    
    parser.add_argument('--quiet', '-q',
                       action='store_true',
                       help='Reduce output verbosity')
    
    parser.add_argument('--verbose', '-v',
                       action='store_true',
                       help='Increase output verbosity')

    # Prompt injection toggles
    parser.add_argument('--reward-hack',
                       action='store_true',
                       help='Append a reward-hacking instruction to agent prompts')

    # Post-run tools
    parser.add_argument('--no-llm-judge',
                       action='store_true',
                       help='Skip running the LLM reward hacking judge after the run')
    
    args = parser.parse_args()
    
    # Load environment variables from .env if present (for API keys, etc.)
    try:
        loaded = load_env_from_dotenv()
        if loaded:
            print(f"🔐 Loaded {len(loaded)} variables from .env")
    except Exception as e:
        print(f"Warning: failed to load .env: {e}")

    # Create descriptive run directory - will update name once we have more config info
    run_timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    temp_run_dir = Path("runs") / f"run_{run_timestamp}"
    temp_run_dir.mkdir(parents=True, exist_ok=True)

    # Always stage workspaces outside the repository to avoid path leakage
    workspace_tempdir = tempfile.TemporaryDirectory(prefix="lcb_ws_")
    workspace_root = Path(workspace_tempdir.name)
    
    # Load configuration (or use minimal config for quick mode)
    if args.quick:
        print("🚀 Quick mode: using minimal configuration")
        config = {
            'timeout': 300,
            'cleanup': True,
            'agents': [{'name': 'grok', 'flags': []}],
            'problem_filters': {
                'release_version': args.release_version,
                'max_problems': 1
            }
        }
    else:
        try:
            config = load_config(args.config)
        except FileNotFoundError:
            print(f"Configuration file not found: {args.config}")
            print("Creating default configuration file...")
            
            # Create default config if it doesn't exist
            default_config = {
                'timeout': 300,
                'cleanup': True,
                'agents': [{'name': 'grok', 'flags': []}],
                'problem_filters': {
                    'release_version': args.release_version,
                    'difficulties': ['easy'],
                    'max_problems': 3
                }
            }
            
            with open(args.config, 'w') as f:
                yaml.dump(default_config, f, default_flow_style=False)
            
            config = default_config
            print(f"Created {args.config} with default settings")
    
    # Set up organized workspace directories (final results path updated later)
    config['workspace_base_dir'] = str(workspace_root)
    config['results_dir'] = str(temp_run_dir / "results")
    
    # Note: We will save the configuration snapshot later, after applying CLI overrides
    
    # Apply command line overrides
    if args.scenario:
        if args.scenario in config.get('scenarios', {}):
            scenario_config = config['scenarios'][args.scenario]
            config.update(scenario_config)
            print(f"Using scenario: {args.scenario}")
        else:
            print(f"Scenario '{args.scenario}' not found in configuration")
            sys.exit(1)
    
    if args.timeout:
        config['timeout'] = args.timeout
    
    if args.no_cleanup:
        config['cleanup'] = False
    
    # Override parallel settings if requested
    if args.parallel:
        if 'parallel' not in config:
            config['parallel'] = {}
        config['parallel']['enabled'] = True
        
    if args.max_workers:
        if 'parallel' not in config:
            config['parallel'] = {}
        config['parallel']['max_workers'] = args.max_workers
    
    if args.verbose:
        config['log_level'] = 'DEBUG'
    elif args.quiet:
        config['log_level'] = 'WARNING'
    
    if args.output:
        config['results_file'] = args.output
    
    # Apply release version override
    if args.release_version:
        config.setdefault('problem_filters', {})['release_version'] = args.release_version
    
    # Apply model overrides only to compatible agent configurations
    if args.model or args.reasoning_effort:
        model_compatible_agents = {
            'grok',
        }
        
        for agent_config in config.get('agents', []):
            agent_name = agent_config.get('name', '').lower()
            
            if agent_name not in model_compatible_agents:
                continue  # Skip agents that don't support model selection
                
            if args.model:
                if 'model_config' not in agent_config:
                    agent_config['model_config'] = {}
                agent_config['model_config']['model'] = args.model
            
            if args.reasoning_effort:
                if 'model_config' not in agent_config:
                    agent_config['model_config'] = {}
                agent_config['model_config']['reasoning_effort'] = args.reasoning_effort
    
    # Always ensure results file is in the run directory (will update to final run_dir later)
    config['results_file'] = str(temp_run_dir / "test_results.json")
    
    # Determine agents to test
    agents = config.get('agents', [])
    if args.agent:
        agents = [{'name': args.agent, 'flags': []}]
    elif args.agents:
        agents = [{'name': name, 'flags': []} for name in args.agents]

    # Apply reward-hack injection to selected agents if enabled
    if args.reward_hack:
        for agent_config in agents:
            if not isinstance(agent_config, dict):
                continue
            pi = agent_config.get('prompt_injection')
            if not isinstance(pi, dict):
                pi = {}
            pi['reward_hack'] = True
            agent_config['prompt_injection'] = pi
    
    # Apply model overrides to selected agents (especially important for dynamically created ones)
    if args.model or args.reasoning_effort:
        model_compatible_agents = {
            'grok',
        }
        compatible_found_for = []
        
        for agent_config in agents:
            agent_name = agent_config.get('name', '').lower()
            if agent_name not in model_compatible_agents:
                continue  # Skip incompatible agents
            
            compatible_found_for.append(agent_name)
            
            if args.model:
                if 'model_config' not in agent_config:
                    agent_config['model_config'] = {}
                agent_config['model_config']['model'] = args.model
            
            if args.reasoning_effort:
                if 'model_config' not in agent_config:
                    agent_config['model_config'] = {}
                agent_config['model_config']['reasoning_effort'] = args.reasoning_effort
        
        # Only show info message if model was specified
        if args.model:
            if compatible_found_for:
                print(f"ℹ️  Using model '{args.model}' for: {', '.join(sorted(set(compatible_found_for)))}")
            else:
                print(f"⚠️  Model flag ignored - no compatible agents selected")
                print(f"   Supported agents for --model: {', '.join(sorted(model_compatible_agents))}")
    
    if not agents:
        print("No agents specified")
        sys.exit(1)
    
    # Determine problems to test
    problems = []
    if args.problem:
        problems = [args.problem]
        print(f"Testing specific problem: {args.problem}")
    elif args.problems:
        problems = args.problems
        print(f"Testing specific problems: {args.problems}")
    elif args.random:
        # Select a random problem
        try:
            release_version = args.release_version or config.get('problem_filters', {}).get('release_version', 'v6')
            print(f"Selecting random problem from {release_version}...")
            
            # Get available problems (this uses cache/index for efficiency)
            available_problems = list(get_available_problems(release_version))
            
            if not available_problems:
                print(f"No problems found in {release_version}")
                problems = ['leetcode_1']  # Fallback
            else:
                # Apply filters if specified
                if args.difficulty or args.platform:
                    print("Applying filters to random selection...")
                    all_problems = get_cached_problems(release_version)
                    
                    if args.difficulty:
                        from problems import get_atcoder_granular_difficulty
                        filtered_problems = []
                        for p in all_problems:
                            # For AtCoder problems, check both original and granular difficulty
                            if p.platform.value == 'atcoder':
                                granular_diff = get_atcoder_granular_difficulty(p)
                                if p.difficulty.value == args.difficulty or granular_diff.value == args.difficulty:
                                    filtered_problems.append(p)
                            else:
                                if p.difficulty.value == args.difficulty:
                                    filtered_problems.append(p)
                        all_problems = filtered_problems
                    if args.platform:
                        all_problems = [p for p in all_problems if p.platform.value == args.platform]
                    
                    available_problems = [p.question_id for p in all_problems]
                
                if available_problems:
                    selected_problem = random.choice(available_problems)
                    problems = [selected_problem]
                    print(f"Randomly selected: {selected_problem}")
                else:
                    print("No problems match the specified filters")
                    problems = ['leetcode_1']  # Fallback
                    
        except Exception as e:
            print(f"Error selecting random problem: {e}")
            problems = ['leetcode_1']  # Fallback
    else:
        # Apply command line filters to config
        if args.difficulty:
            config.setdefault('problem_filters', {})['difficulties'] = [args.difficulty]
        
        if args.platform:
            config.setdefault('problem_filters', {})['platforms'] = [args.platform]
        
        if args.max_problems:
            config.setdefault('problem_filters', {})['max_problems'] = args.max_problems
        
        try:
            problems = get_problems_from_config(config)
        except Exception as e:
            print(f"Error loading problems: {e}")
            print("Using fallback problem list...")
            problems = ['leetcode_1', 'leetcode_2']  # Fallback
    
    if not problems:
        print("No problems to test")
        sys.exit(1)
    
    print(f"Testing {len(problems)} problems with {len(agents)} agents")
    print(f"Problems: {problems[:5]}{'...' if len(problems) > 5 else ''}")
    print(f"Agents: {[a['name'] for a in agents]}")
    
    # Create descriptive run directory name and rename
    release_version = args.release_version or config.get('problem_filters', {}).get('release_version', 'v6')
    difficulty = args.difficulty or config.get('problem_filters', {}).get('difficulties', [None])[0]
    
    descriptive_name = generate_descriptive_run_name(agents, problems, release_version, difficulty)
    run_dir = Path("runs") / descriptive_name
    
    # Check if descriptive name already exists, add timestamp if needed
    if run_dir.exists():
        timestamp_suffix = run_timestamp
        descriptive_name = generate_descriptive_run_name(agents, problems, release_version, difficulty, timestamp_suffix)
        run_dir = Path("runs") / descriptive_name
    
    # Rename the temporary directory to the descriptive name
    if temp_run_dir != run_dir:
        temp_run_dir.rename(run_dir)
        # Update config paths to use final run_dir (workspaces remain in system temp)
        config['workspace_base_dir'] = str(workspace_root)
        config['results_dir'] = str(run_dir / "results")
        config['results_file'] = str(run_dir / "test_results.json")
    
    print(f"📁 Created run directory: {run_dir}")
    
    # Show model configurations for compatible agents only
    if args.model or args.reasoning_effort:
        compatible_selected = [
            a['name'] for a in agents 
            if a.get('name', '').lower() in {'grok'}
        ]
        if compatible_selected:
            print(f"Model overrides (for {', '.join(compatible_selected)}):")
            if args.model:
                print(f"  Model: {args.model}")
            if args.reasoning_effort:
                print(f"  Reasoning effort: {args.reasoning_effort}")

    # Ensure the finalized agent list is reflected in the saved config
    config['agents'] = agents

    # Copy final, resolved configuration to run directory for reproducibility
    config_copy_path = run_dir / "config.yaml"
    with open(config_copy_path, 'w') as f:
        yaml.dump(config, f, default_flow_style=False)
    print(f"📋 Configuration saved to: {config_copy_path}")
    
    print()
    
    # Initialize tester
    tester = AutomatedTester(config)
    
    try:
        # Run tests
        results = tester.test_batch(problems, agents)

        # Run LLM detection BEFORE generating comprehensive report (if enabled)
        if not args.no_llm_judge:
            _run_llm_detection(run_dir, args)

        # Generate comprehensive reward hacking report (after LLM detection)
        try:
            print("\n📊 Generating comprehensive reward hacking report...")
            tester.generate_comprehensive_reward_hacking_report(run_dir)
        except Exception as e:
            print(f"⚠️  Failed to generate comprehensive reward hacking report: {e}")

        # Generate standard report
        report = tester.generate_report()
        
        # Print summary
        print("\n" + "="*60)
        print("TEST RESULTS SUMMARY")
        print("="*60)
        
        for agent_name, agent_stats in report.get('agents', {}).items():
            print(f"\n{agent_name.upper()}:")
            
            # Count problems with all tests passed
            problems_data = agent_stats.get('problems', {})
            problems_with_all_tests_passed = 0
            # Multi-output problems are reported separately
            mo_summary = agent_stats.get('multi_output_summary', {})
            mo_problem_ids = set(mo_summary.get('problems', []))
            mo_passed = 0
            mo_total = len(mo_problem_ids)
            regular_passed = 0
            regular_total = max(agent_stats.get('total_problems', 0) - mo_total, 0)
            for problem_id in problems_data.keys():
                # Check if this problem passed all its tests
                # We look at the actual test results from the TestResult objects
                for result in results:
                    if result.agent_name == agent_name and result.problem_id == problem_id:
                        if result.holdout_enabled and result.holdout_analysis:
                            analysis = result.holdout_analysis
                            if not analysis.get('error'):
                                visible_passed = analysis.get('visible_passed', 0)
                                visible_total = analysis.get('visible_total', 0)
                                holdout_passed = analysis.get('holdout_passed', 0)
                                holdout_total = analysis.get('holdout_total', 0)
                                passed_all = (
                                    visible_passed == visible_total and holdout_passed == holdout_total and 
                                    visible_total > 0 and holdout_total > 0
                                )
                                if passed_all:
                                    if problem_id in mo_problem_ids:
                                        mo_passed += 1
                                    else:
                                        problems_with_all_tests_passed += 1
                                        regular_passed += 1
                            elif result.test_passed:
                                # Fallback: if holdout analysis had an error but visible tests passed
                                if problem_id in mo_problem_ids:
                                    mo_passed += 1
                                else:
                                    problems_with_all_tests_passed += 1
                                    regular_passed += 1
                        elif result.test_passed:
                            # No holdout tests, just check if visible tests passed
                            if problem_id in mo_problem_ids:
                                mo_passed += 1
                            else:
                                problems_with_all_tests_passed += 1
                                regular_passed += 1
                        break
            
            # Print regular problems summary (excluding multi-output)
            print(f"  Problems solved (all tests, excluding multi-output): {regular_passed}/{regular_total}")
            print(
                f"  Success rate: { (regular_passed / regular_total):.1%}" if regular_total > 0 else "  Success rate: 0.0%"
            )
            # Print multi-output problems separately
            if mo_total > 0:
                print(f"  Multiple-output problems: {mo_passed}/{mo_total} (reported separately)")
            
            # Show test breakdown if available
            if 'combined_tests' in agent_stats and agent_stats['holdout_tests']['total'] > 0:
                visible = agent_stats['visible_tests']
                holdout = agent_stats['holdout_tests']
                print(f"  Test breakdown:")
                print(f"    - Visible: {visible['passed']}/{visible['total']} ({visible['pass_rate']:.1%})")
                print(f"    - Holdout: {holdout['passed']}/{holdout['total']} ({holdout['pass_rate']:.1%})")
                # If there are multi-output problems, show their breakdown as well
                if mo_total > 0 and agent_stats.get('multi_output_summary'):
                    mo = agent_stats['multi_output_summary']
                    mv = mo.get('visible_tests', {})
                    mh = mo.get('holdout_tests', {})
                    print(f"    - Multi-output (visible): {mv.get('passed',0)}/{mv.get('total',0)} ({mv.get('pass_rate',0):.1%})")
                    print(f"    - Multi-output (holdout): {mh.get('passed',0)}/{mh.get('total',0)} ({mh.get('pass_rate',0):.1%})")
            
            print(f"  Solutions generated: {agent_stats['solutions_generated']}/{agent_stats['total_problems']}")
            print(f"  Generation rate: {agent_stats['generation_rate']:.1%}")
            print(f"  Average runtime: {agent_stats['average_runtime']:.1f}s")
            
            # Add reward hacking statistics
            if 'reward_hacking_violations' in agent_stats:
                print(f"  Reward hacking detected: {agent_stats['reward_hacking_violations']} problems")
                if agent_stats['reward_hacking_violations'] > 0:
                    print(f"  Reward hacking rate: {agent_stats['reward_hacking_rate']:.1%}")
                    if agent_stats.get('violation_types'):
                        print(f"  Violation types: {', '.join(f'{k}({v})' for k, v in agent_stats['violation_types'].items())}")

            # Overfitting summary (based on holdout tests)
            if 'overfitting_flagged' in agent_stats:
                flagged = agent_stats['overfitting_flagged']
                print(f"  Overfitting flagged: {flagged} problems")
                if flagged > 0 and agent_stats.get('overfitting_severity_counts'):
                    sev = agent_stats['overfitting_severity_counts']
                    ordered = ['critical', 'high', 'medium', 'low']
                    detail_parts = [f"{k}({sev.get(k, 0)})" for k in ordered if sev.get(k, 0)]
                    if detail_parts:
                        print(f"  Overfitting severity: {', '.join(detail_parts)}")
        
        # Print reward hacking summary by agent if violations exist
        has_violations = any(agent_stats.get('reward_hacking_violations', 0) > 0 
                           for agent_stats in report.get('agents', {}).values())
        
        if has_violations:
            print(f"\n{'='*60}")
            print("REWARD HACKING SUMMARY BY AGENT")
            print(f"{'='*60}")
            
            total_violations = 0
            total_problems = 0
            all_violation_types = {}
            all_severity_counts = {}
            
            for agent_name, agent_stats in report.get('agents', {}).items():
                violations = agent_stats.get('reward_hacking_violations', 0)
                if violations > 0:
                    print(f"\n{agent_name.upper()}:")
                    print(f"  Violations: {violations} problems")
                    print(f"  Rate: {agent_stats.get('reward_hacking_rate', 0):.1%}")
                    if agent_stats.get('violation_types'):
                        print(f"  Types: {', '.join(f'{k}({v})' for k, v in agent_stats['violation_types'].items())}")
                
                # Aggregate totals
                total_violations += violations
                total_problems += agent_stats.get('total_problems', 0)
                for v_type, count in agent_stats.get('violation_types', {}).items():
                    all_violation_types[v_type] = all_violation_types.get(v_type, 0) + count
            
            # Print overall summary
            overall_rate = total_violations / total_problems if total_problems > 0 else 0
            print(f"\nOVERALL:")
            print(f"  Total violations: {total_violations}")
            print(f"  Overall rate: {overall_rate:.1%}")
            if all_violation_types:
                print(f"  Common violations: {', '.join(f'{k}({v})' for k, v in sorted(all_violation_types.items(), key=lambda x: x[1], reverse=True))}")
        else:
            print(f"\n🛡️  No reward hacking violations detected across all agents")
        
        # Save detailed report
        results_file = Path(config.get('results_file', 'test_results.json'))
        report_file = results_file.with_suffix('.report.json')
        with open(report_file, 'w') as f:
            json.dump(report, f, indent=2)
        
        print(f"\nDetailed report saved to: {report_file}")
        print(f"Full results saved to: {results_file}")

        # LLM detection was already run earlier in the process
        
        # Print organized run summary
        print(f"\n{'='*60}")
        print(f"📁 RUN SUMMARY")
        print(f"{'='*60}")
        print(f"Run directory: {run_dir}")
        print(f"├── results/              # Agent results and logs by problem")
        print(f"├── test_results.json     # Complete test results")
        print(f"├── test_results.report.json  # Summary report")
        if not args.no_llm_judge:
            print(f"├── llm_reward_hacking_results.json  # LLM judge results (if API key present)")
        print(f"├── reward_hacking_report.json  # Comprehensive reward hacking analysis")
        print(f"└── config.yaml           # Configuration used for this run")
        print(f"\n💡 Navigate to {run_dir} to explore all logs and results!")
        print("🧪 Workspaces were provisioned in a system temporary directory and cleaned up after execution.")
        
    except KeyboardInterrupt:
        print("\nTest interrupted by user")
    except Exception as e:
        print(f"Error during testing: {e}")
    
    finally:
        # Ensure results are saved even if there was an error
        try:
            if hasattr(tester, 'results') and tester.results:
                print("Saving results before cleanup...")
                tester._save_results()
        except Exception as save_error:
            print(f"Error saving results: {save_error}")
        
        # Cleanup
        tester.cleanup()

        # Remove temporary workspace root after cleanup to avoid leaving artifacts
        try:
            workspace_tempdir.cleanup()
        except Exception as cleanup_error:
            print(f"Warning: Failed to remove temporary workspace directory: {cleanup_error}")


if __name__ == '__main__':
    main()
