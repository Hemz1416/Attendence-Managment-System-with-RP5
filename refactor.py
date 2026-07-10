import os
import re
from pathlib import Path

def process_file(filepath):
    with open(filepath, 'r', encoding='utf-8') as f:
        content = f.read()

    # 1. Remove sys.path manipulation
    sys_path_pattern = re.compile(
        r"app_dir = Path\(__file__\)\.resolve\(\)\.parent(?:\.parent)?\nif str\(app_dir\) not in sys\.path:\n\s+sys\.path\.insert\(0,\s*str\(app_dir\)\)\n",
        re.MULTILINE
    )
    content = sys_path_pattern.sub("", content)

    # 2. Refactor imports
    content = re.sub(r"^import config", "from app import config", content, flags=re.MULTILINE)
    content = re.sub(r"^import database", "from app import database", content, flags=re.MULTILINE)
    content = re.sub(r"^import utils", "from app import utils", content, flags=re.MULTILINE)
    content = re.sub(r"^import cleanup_enrollment_images", "from app import cleanup_enrollment_images", content, flags=re.MULTILINE)
    content = re.sub(r"^from attendance_login import", "from app.attendance_login import", content, flags=re.MULTILINE)

    # 3. Add import logging if print is used
    if 'print(' in content and 'import logging' not in content:
        # Add import logging after other standard imports
        content = re.sub(r"^(import .*)$", r"\1\nimport logging", content, count=1, flags=re.MULTILINE)

    # 4. Replace print with logging
    def replace_print(match):
        text = match.group(1)
        if '[Warning]' in text or '[WARNING]' in text:
            text = text.replace('[Warning]', '').replace('[WARNING]', '').strip()
            return f"logging.warning({text})"
        elif '[Error]' in text or '[ERROR]' in text:
            text = text.replace('[Error]', '').replace('[ERROR]', '').strip()
            return f"logging.error({text})"
        else:
            return f"logging.info({text})"

    content = re.sub(r"print\((.*?)\)", replace_print, content)

    with open(filepath, 'w', encoding='utf-8') as f:
        f.write(content)

def main():
    root = Path(r"h:\Internship Siri AB\Project_Code\RP5")
    app_dir = root / "app"
    gui_dir = app_dir / "gui"

    # Create __init__.py
    (app_dir / "__init__.py").touch(exist_ok=True)
    (gui_dir / "__init__.py").touch(exist_ok=True)

    # Process all py files
    for root_dir, dirs, files in os.walk(app_dir):
        for file in files:
            if file.endswith(".py") and file != "__init__.py":
                process_file(os.path.join(root_dir, file))
                
    # Also process main.py and reset_deployment.py
    process_file(root / "main.py")
    process_file(root / "reset_deployment.py")
    process_file(root / "validate_deployment.py")

if __name__ == "__main__":
    main()
