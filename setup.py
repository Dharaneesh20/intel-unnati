"""
Setup configuration for Intel AI Agent Framework.
"""

from setuptools import setup, find_packages
from pathlib import Path

# Read README for long description
this_directory = Path(__file__).parent
long_description = (this_directory / "README.md").read_text(encoding="utf-8")

# Read requirements
requirements = (this_directory / "requirements.txt").read_text().splitlines()
dev_requirements = (this_directory / "requirements-dev.txt").read_text().splitlines()

setup(
    name="intel-ai-agent-framework",
    version="1.0.0",
    author="Intel AI Agent Framework Team",
    author_email="team@intel-ai-framework.com",
    description="Production-ready AI Agent framework for orchestrating complex agentic workflows",
    long_description=long_description,
    long_description_content_type="text/markdown",
    url="https://github.com/Dharaneesh20/intel-unnati",
    project_urls={
        "Documentation": "https://dharaneesh20.github.io/intel-unnati/",
        "Source": "https://github.com/Dharaneesh20/intel-unnati",
        "Bug Reports": "https://github.com/Dharaneesh20/intel-unnati/issues",
    },
    packages=find_packages(where="src"),
    package_dir={"": "src"},
    classifiers=[
        "Development Status :: 4 - Beta",
        "Intended Audience :: Developers",
        "License :: OSI Approved :: MIT License",
        "Operating System :: OS Independent",
        "Programming Language :: Python :: 3",
        "Programming Language :: Python :: 3.9",
        "Programming Language :: Python :: 3.10",
        "Programming Language :: Python :: 3.11",
        "Topic :: Software Development :: Libraries :: Python Modules",
        "Topic :: Scientific/Engineering :: Artificial Intelligence",
    ],
    python_requires=">=3.9",
    install_requires=requirements,
    extras_require={
        "dev": dev_requirements,
        "intel": [
            "openvino>=2023.1.0",
            "intel-extension-for-pytorch>=2.1.0",
        ],
        "apache": [
            "apache-airflow>=2.7.0",
            "kafka-python>=2.0.2",
        ],
        "all": dev_requirements + [
            "openvino>=2023.1.0",
            "intel-extension-for-pytorch>=2.1.0",
            "apache-airflow>=2.7.0",
            "kafka-python>=2.0.2",
        ]
    },
    entry_points={
        "console_scripts": [
            "intel-ai-framework=src.cli:main",
        ],
    },
    include_package_data=True,
    zip_safe=False,
    keywords="ai, agents, workflow, automation, intel, apache, machine-learning",
)
