#!/bin/bash
# Repository Cleanup Script for BP-face-recognition
# Run this to clean temporary files before committing

echo "==================================="
echo "Repository Cleanup Script"
echo "==================================="

# Colors for output
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
NC='\033[0m' # No Color

echo ""
echo -e "${YELLOW}Step 1: Cleaning training logs...${NC}"
rm -f training_option_*.log
rm -f *.pid
rm -f *.log
echo -e "${GREEN}✓ Training logs cleaned${NC}"

echo ""
echo -e "${YELLOW}Step 2: Cleaning Python cache...${NC}"
# Remove __pycache__ directories (be careful with this)
find . -type d -name "__pycache__" -exec rm -rf {} + 2>/dev/null || true
# Remove compiled Python files
find . -name "*.pyc" -delete 2>/dev/null || true
find . -name "*.pyo" -delete 2>/dev/null || true
# Remove backup files
find . -name "*~" -delete 2>/dev/null || true
find . -name "*.bak" -delete 2>/dev/null || true
echo -e "${GREEN}✓ Python cache cleaned${NC}"

echo ""
echo -e "${YELLOW}Step 3: Organizing temporary documentation...${NC}"
# Move SESSION_14_SUMMARY.md to .maintenance/
if [ -f "SESSION_14_SUMMARY.md" ]; then
    mv SESSION_14_SUMMARY.md .maintenance/
    echo -e "${GREEN}✓ Moved SESSION_14_SUMMARY.md to .maintenance/${NC}"
fi

echo ""
echo -e "${YELLOW}Step 4: Checking for other temp files...${NC}"
# List any remaining .tmp files
TMP_FILES=$(find . -name "*.tmp" 2>/dev/null | wc -l)
if [ "$TMP_FILES" -gt 0 ]; then
    echo -e "${YELLOW}Found $TMP_FILES .tmp files${NC}"
    find . -name "*.tmp" -delete 2>/dev/null || true
    echo -e "${GREEN}✓ Removed .tmp files${NC}"
else
    echo -e "${GREEN}✓ No .tmp files found${NC}"
fi

echo ""
echo "==================================="
echo -e "${GREEN}Cleanup Complete!${NC}"
echo "==================================="
echo ""
echo "Files removed:"
echo "  - Training logs (*.log)"
echo "  - Process IDs (*.pid)"
echo "  - Python cache (__pycache__, *.pyc, *.pyo)"
echo "  - Backup files (*~, *.bak)"
echo "  - Temporary files (*.tmp)"
echo ""
echo "Files moved:"
echo "  - SESSION_14_SUMMARY.md → .maintenance/"
echo ""
echo "Repository is now clean and ready for commits!"
