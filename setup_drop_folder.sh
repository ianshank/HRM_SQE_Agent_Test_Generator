#!/bin/bash
# Setup script for Drop Folder Test Generation System

set -e  # Exit on error

echo "============================================"
echo "Drop Folder Setup Script"
echo "============================================"
echo ""

# Colors for output
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
NC='\033[0m' # No Color

# Get script directory
SCRIPT_DIR="$( cd "$( dirname "${BASH_SOURCE[0]}" )" && pwd )"
cd "$SCRIPT_DIR"

echo "Step 1: Checking Python installation..."
if ! command -v python3 &> /dev/null; then
    echo "Error: Python 3 is not installed"
    exit 1
fi
echo -e "${GREEN}✓${NC} Python 3 found: $(python3 --version)"
echo ""

echo "Step 2: Installing watchdog library..."
pip3 install watchdog>=3.0.0 --quiet || {
    echo -e "${YELLOW}Warning: Could not install watchdog. Polling mode will be used.${NC}"
}
echo -e "${GREEN}✓${NC} Dependencies checked"
echo ""

echo "Step 3: Creating drop folder structure..."
python3 -m hrm_eval.drop_folder setup
echo -e "${GREEN}✓${NC} Folder structure created"
echo ""

echo "Step 4: Setting permissions..."
chmod -R u+rw drop_folder/ 2>/dev/null || true
echo -e "${GREEN}✓${NC} Permissions set"
echo ""

echo "Step 5: Creating sample requirement file..."
cat > drop_folder/input/example_requirement.txt << 'EOF'
Epic: User Profile Management

User Story 1: Update Profile Information
As a registered user, I want to update my profile information
So that my account details remain current

Acceptance Criteria:
- User can edit name, email, and phone number
- Changes are validated before saving
- Confirmation message shown after successful update
- Email changes require verification

User Story 2: Upload Profile Picture
As a registered user, I want to upload a profile picture
So that others can recognize me

Acceptance Criteria:
- Supported formats: JPG, PNG, GIF
- Maximum file size: 5MB
- Image preview shown before upload
- Old picture is replaced after successful upload
EOF
echo -e "${GREEN}✓${NC} Sample file created: drop_folder/input/example_requirement.txt"
echo ""

echo "============================================"
echo "Setup Complete!"
echo "============================================"
echo ""
echo "Next steps:"
echo ""
echo "1. View the sample requirement:"
echo "   cat drop_folder/input/example_requirement.txt"
echo ""
echo "2. Process the sample file:"
echo "   python3 -m hrm_eval.drop_folder process"
echo ""
echo "3. Or start watching for new files:"
echo "   python3 -m hrm_eval.drop_folder watch"
echo ""
echo "4. Check the generated tests:"
echo "   ls -la drop_folder/output/"
echo ""
echo "For more information, see: drop_folder/README.md"
echo ""
