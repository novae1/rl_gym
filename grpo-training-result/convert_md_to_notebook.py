import json
import re
from pathlib import Path

def parse_markdown_to_notebook(md_file_path, output_path):
    """Convert markdown file to Jupyter notebook."""

    with open(md_file_path, 'r', encoding='utf-8') as f:
        content = f.read()

    # Initialize notebook structure
    notebook = {
        "cells": [],
        "metadata": {
            "kernelspec": {
                "display_name": "Python 3",
                "language": "python",
                "name": "python3"
            },
            "language_info": {
                "codemirror_mode": {
                    "name": "ipython",
                    "version": 3
                },
                "file_extension": ".py",
                "mimetype": "text/x-python",
                "name": "python",
                "nbconvert_exporter": "python",
                "pygments_lexer": "ipython3",
                "version": "3.10.0"
            }
        },
        "nbformat": 4,
        "nbformat_minor": 4
    }

    # Split content into lines for processing
    lines = content.split('\n')
    i = 0

    while i < len(lines):
        line = lines[i]

        # Check if this is a title line (starts with #)
        if line.strip().startswith('#') and line.strip() != '---':
            # Extract title
            title = line
            i += 1

            # Add title cell
            notebook["cells"].append({
                "cell_type": "markdown",
                "metadata": {},
                "source": [title]
            })

            # Collect content until we hit a code block or another title
            content_lines = []
            while i < len(lines):
                if lines[i].strip().startswith('#') and lines[i].strip() != '---':
                    # Next title found, break
                    break
                elif lines[i].strip().startswith('```python'):
                    # Code block found, break
                    break
                else:
                    content_lines.append(lines[i])
                    i += 1

            # Add content cell (if there's content)
            content_text = '\n'.join(content_lines).strip()
            if content_text and content_text != '---':
                notebook["cells"].append({
                    "cell_type": "markdown",
                    "metadata": {},
                    "source": [content_text]
                })

            # Check if there's a code block
            if i < len(lines) and lines[i].strip().startswith('```python'):
                i += 1  # Skip the ```python line

                # Collect code lines
                code_lines = []
                while i < len(lines) and not lines[i].strip().startswith('```'):
                    code_lines.append(lines[i])
                    i += 1

                # Skip the closing ```
                if i < len(lines):
                    i += 1

                code_text = '\n'.join(code_lines)

                # Special handling for Section 7 (evaluation)
                if '# 7. Avaliação no split de teste' in title or 'NUM_SAMPLES = 1319' in code_text:
                    # Split into 4 logical parts
                    split_section_7_code(code_text, notebook)
                else:
                    # Add regular code cell
                    notebook["cells"].append({
                        "cell_type": "code",
                        "execution_count": None,
                        "metadata": {},
                        "outputs": [],
                        "source": [code_text]
                    })
        else:
            i += 1

    # Write notebook to file
    with open(output_path, 'w', encoding='utf-8') as f:
        json.dump(notebook, f, ensure_ascii=False, indent=2)

    print(f"Notebook created successfully: {output_path}")


def split_section_7_code(code_text, notebook):
    """Split Section 7 evaluation code into 4 logical parts."""

    # Part 1: Configuration and imports (up to but not including def load_base_model)
    part1_pattern = r'^(.*?)(?=def load_base_model)'
    part1_match = re.search(part1_pattern, code_text, re.DOTALL)
    if part1_match:
        part1 = part1_match.group(1).strip()
        notebook["cells"].append({
            "cell_type": "code",
            "execution_count": None,
            "metadata": {},
            "outputs": [],
            "source": [part1]
        })

    # Part 2: Helper functions (load_base_model through evaluate_answer)
    part2_pattern = r'(def load_base_model\(\):.*?def evaluate_answer\(extracted, truth\):.*?return str\(extracted\)\.strip\(\) == str\(truth\)\.strip\(\))'
    part2_match = re.search(part2_pattern, code_text, re.DOTALL)
    if part2_match:
        part2 = part2_match.group(1).strip()
        notebook["cells"].append({
            "cell_type": "code",
            "execution_count": None,
            "metadata": {},
            "outputs": [],
            "source": [part2]
        })

    # Part 3: Generation and evaluation functions
    part3_pattern = r'(def generate_batch\(.*?\n    return \{"accuracy": acc.*?\})'
    part3_match = re.search(part3_pattern, code_text, re.DOTALL)
    if part3_match:
        part3 = part3_match.group(1).strip()
        notebook["cells"].append({
            "cell_type": "code",
            "execution_count": None,
            "metadata": {},
            "outputs": [],
            "source": [part3]
        })

    # Part 4: Main execution (if RUN_EVAL block)
    part4_pattern = r'(if RUN_EVAL:.*?)$'
    part4_match = re.search(part4_pattern, code_text, re.DOTALL)
    if part4_match:
        part4 = part4_match.group(1).strip()
        notebook["cells"].append({
            "cell_type": "code",
            "execution_count": None,
            "metadata": {},
            "outputs": [],
            "source": [part4]
        })


if __name__ == "__main__":
    md_file = Path(r"c:\Users\novae\Documents\rl_gym\grpo-training-result\GRPO_Gemma3_GSM8K.md")
    output_file = Path(r"c:\Users\novae\Documents\rl_gym\grpo-training-result\GRPO_Gemma3_GSM8K.ipynb")

    parse_markdown_to_notebook(md_file, output_file)
