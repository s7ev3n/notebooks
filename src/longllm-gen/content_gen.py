import os

from .prompt_template import prompt_bite_content


system_prompt = prompt_bite_content.format(
    knowledge="\n\n".join(file_content)
)
