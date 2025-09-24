#!/bin/bash

# Build script for collaborative-gym package

echo "Building collaborative-gym package..."

# Clean previous builds
rm -rf build/
rm -rf dist/
rm -rf *.egg-info/

# Install build dependencies
pip install --upgrade pip setuptools wheel twine build

# Build the package
python -m build

# Check the package
twine check dist/*

echo "Package built successfully!"
echo "Files created:"
ls -la dist/

echo ""
echo "To upload to TestPyPI (for testing):"
echo "twine upload --repository testpypi dist/*"
echo ""
echo "To upload to PyPI (production):"
echo "twine upload dist/*"