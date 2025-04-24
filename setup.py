import setuptools

if __name__ == "__main__":
    with open("README.md", "r", encoding="utf-8") as fh:
        long_description = fh.read()

    setuptools.setup(
        author_email="Clement.Grisi@radboudumc.nl",
        long_description=long_description,
        long_description_content_type="text/markdown",
        url="https://github.com/DIAGNijmegen/unicorn_baseline",
        project_urls={
            "Bug Tracker": "https://github.com/DIAGNijmegen/unicorn_baseline/issues"
        },
        package_dir={"": "src"},  # our packages live under src, but src is not a package itself
        packages=setuptools.find_packages("src", exclude=["tests"]),
        exclude_package_data={"": ["tests"]},
        entry_points={
            "console_scripts": [
                "unicorn_baseline = unicorn_baseline.inference:run",
            ],
        },
    )
