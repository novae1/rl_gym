"""
Test suite for reward functions in utils.py
Tests both legacy <SOLUTION> pattern and new $\boxed{...}$ pattern
"""

import sys
sys.path.insert(0, '/workspace/rl_gym/grpo-training-result')

from utils import check_numbers, check_answer, match_format_exactly, match_format_approximately

def test_check_numbers():
    """Test the check_numbers function with various input patterns"""

    print("=" * 60)
    print("Testing check_numbers() function")
    print("=" * 60)

    # Test cases: (description, response, expected_extraction, expected_score)
    test_cases = [
        # Original SOLUTION pattern tests
        (
            "Old format: SOLUTION with correct answer",
            "<start_working_out>Work here<end_working_out><SOLUTION>476</SOLUTION>",
            "476",
            1.5  # Correct answer
        ),
        (
            "Old format: SOLUTION with wrong answer",
            "<start_working_out>Work here<end_working_out><SOLUTION>100</SOLUTION>",
            "100",
            0.0  # Wrong answer
        ),
        (
            "Old format: SOLUTION with verbose answer",
            "<start_working_out>Work here<end_working_out><SOLUTION>The answer is 476</SOLUTION>",
            "476",
            1.5  # Should extract first number
        ),

        # New boxed pattern tests
        (
            "New format: Single boxed answer (correct)",
            "After working out the problem:\nFinal Answer: The final answer is $\\boxed{476}$",
            "476",
            1.5  # Correct answer
        ),
        (
            "New format: Single boxed answer (close to correct)",
            "After working out the problem:\nFinal Answer: The final answer is $\\boxed{475.80}$",
            "475.80",
            0.0  # Close but not exact (476 vs 475.80)
        ),
        (
            "New format: Multiple boxed values (should take LAST)",
            "Step 1: $\\boxed{100}$ intermediate result\nStep 2: $\\boxed{200}$ another step\nFinal: $\\boxed{476}$",
            "476",
            1.5  # Should extract last boxed value
        ),
        (
            "New format: Boxed with decimal",
            "Final answer: $\\boxed{476.0}$",
            "476.0",
            1.5  # Should handle decimals
        ),

        # Edge cases
        (
            "No pattern found",
            "Just some text without any pattern",
            None,
            0.0  # No extraction = 0 score
        ),
        (
            "Both patterns present (should prefer boxed)",
            "<SOLUTION>100</SOLUTION>\nBut actually: $\\boxed{476}$",
            "476",
            1.5  # Should prefer boxed pattern
        ),
        (
            "Boxed with text inside",
            "$\\boxed{476 \\ \\text{dollars}}$",
            "476 \\ \\text{dollars}",
            1.5  # Should extract everything inside boxed (will be converted to float)
        ),
    ]

    true_answer = "476"  # The correct answer for all tests

    for i, (description, response, expected_extraction, expected_score) in enumerate(test_cases, 1):
        print(f"\n--- Test {i}: {description} ---")

        # Prepare inputs in the format check_numbers expects
        prompts = [[
            {"role": "system", "content": "system prompt"},
            {"role": "user", "content": "A concert ticket costs $40. Mr. Benson bought 12 tickets..."}
        ]]

        completions = [[{"content": response}]]
        answer = [true_answer]

        # Run the function
        scores = check_numbers(prompts, completions, answer)

        print(f"Response: {response[:80]}{'...' if len(response) > 80 else ''}")
        print(f"Expected score: {expected_score}")
        print(f"Actual score:   {scores[0]}")

        # Verify score matches expectation
        if abs(scores[0] - expected_score) < 0.01:
            print("✅ PASS")
        else:
            print("❌ FAIL")

    print("\n" + "=" * 60)


def test_check_answer():
    """Test the check_answer function to ensure strict format checking still works"""

    print("\n" + "=" * 60)
    print("Testing check_answer() function (strict format)")
    print("=" * 60)

    test_cases = [
        (
            "Perfect format with correct answer",
            "<start_working_out>Working...<end_working_out><SOLUTION>476</SOLUTION>",
            3.0  # Exact match
        ),
        (
            "Perfect format with whitespace",
            "<start_working_out>Working...<end_working_out><SOLUTION> 476 </SOLUTION>",
            1.5  # Whitespace difference
        ),
        (
            "Missing SOLUTION tags (boxed instead)",
            "Working...\nFinal: $\\boxed{476}$",
            0.0  # check_answer is strict - doesn't recognize boxed
        ),
        (
            "Perfect format, wrong answer",
            "<start_working_out>Working...<end_working_out><SOLUTION>100</SOLUTION>",
            -0.5  # Wrong answer penalty
        ),
    ]

    true_answer = "476"

    for i, (description, response, expected_score) in enumerate(test_cases, 1):
        print(f"\n--- Test {i}: {description} ---")

        prompts = [[
            {"role": "system", "content": "system prompt"},
            {"role": "user", "content": "Test question"}
        ]]

        completions = [[{"content": response}]]
        answer = [true_answer]

        scores = check_answer(prompts, completions, answer)

        print(f"Response: {response[:80]}{'...' if len(response) > 80 else ''}")
        print(f"Expected score: {expected_score}")
        print(f"Actual score:   {scores[0]}")

        if abs(scores[0] - expected_score) < 0.01:
            print("✅ PASS")
        else:
            print("❌ FAIL")

    print("\n" + "=" * 60)


def test_format_functions():
    """Test format checking functions"""

    print("\n" + "=" * 60)
    print("Testing format checking functions")
    print("=" * 60)

    # Test exact format matching
    print("\n--- match_format_exactly ---")

    test_cases = [
        (
            "Perfect format",
            "<start_working_out>Work<end_working_out><SOLUTION>476</SOLUTION>",
            3.0
        ),
        (
            "Boxed format (no format tags)",
            "$\\boxed{476}$",
            0.0  # Doesn't match exact format
        ),
    ]

    for description, response, expected_score in test_cases:
        completions = [[{"content": response}]]
        scores = match_format_exactly(completions)

        print(f"{description}: {scores[0]} (expected {expected_score}) ", end="")
        print("✅" if abs(scores[0] - expected_score) < 0.01 else "❌")

    print("\n" + "=" * 60)


if __name__ == "__main__":
    print("\n🧪 Running reward function tests\n")

    # Run all tests
    test_check_numbers()
    test_check_answer()
    test_format_functions()

    print("\n✨ Testing complete!\n")
