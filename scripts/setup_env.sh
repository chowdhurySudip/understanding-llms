#!/usr/bin/env bash
"""
Setup environment script for Linux-based systems.

This script installs:
- uv package manager (Python package manager)
- vim (text editor)
- git (version control system)

Usage:
    bash scripts/setup_env.sh
    # or
    chmod +x scripts/setup_env.sh
    ./scripts/setup_env.sh
"""

set -e  # Exit on error

echo "================================"
echo "Environment Setup Script"
echo "================================"
echo ""

# Detect OS
if [[ "$OSTYPE" == "linux-gnu"* ]]; then
    echo "Detected Linux OS"
    
    # Detect package manager
    if command -v apt-get &> /dev/null; then
        PKG_MGR="apt"
        echo "Using APT package manager"
    elif command -v dnf &> /dev/null; then
        PKG_MGR="dnf"
        echo "Using DNF package manager"
    elif command -v yum &> /dev/null; then
        PKG_MGR="yum"
        echo "Using YUM package manager"
    elif command -v pacman &> /dev/null; then
        PKG_MGR="pacman"
        echo "Using Pacman package manager"
    else
        echo "Error: No supported package manager found (apt, dnf, yum, pacman)"
        exit 1
    fi
elif [[ "$OSTYPE" == "darwin"* ]]; then
    echo "Detected macOS"
    PKG_MGR="brew"
    if ! command -v brew &> /dev/null; then
        echo "Homebrew not found. Installing Homebrew..."
        /bin/bash -c "$(curl -fsSL https://raw.githubusercontent.com/Homebrew/install/HEAD/install.sh)"
    fi
else
    echo "Error: Unsupported OS type: $OSTYPE"
    exit 1
fi

echo ""
echo "================================"
echo "Installing packages..."
echo "================================"
echo ""

# Install vim
echo ">>> Installing vim..."
if command -v vim &> /dev/null; then
    echo "vim is already installed: $(vim --version | head -n 1)"
else
    case $PKG_MGR in
        apt)
            sudo apt-get update
            sudo apt-get install -y vim
            ;;
        dnf)
            sudo dnf install -y vim
            ;;
        yum)
            sudo yum install -y vim
            ;;
        pacman)
            sudo pacman -S --noconfirm vim
            ;;
        brew)
            brew install vim
            ;;
    esac
    echo "vim installed successfully"
fi

echo ""

# Install git
echo ">>> Installing git..."
if command -v git &> /dev/null; then
    echo "git is already installed: $(git --version)"
else
    case $PKG_MGR in
        apt)
            sudo apt-get update
            sudo apt-get install -y git
            ;;
        dnf)
            sudo dnf install -y git
            ;;
        yum)
            sudo yum install -y git
            ;;
        pacman)
            sudo pacman -S --noconfirm git
            ;;
        brew)
            brew install git
            ;;
    esac
    echo "git installed successfully"
fi

echo ""

# Install uv package manager
echo ">>> Installing uv package manager..."
if command -v uv &> /dev/null; then
    echo "uv is already installed: $(uv --version)"
else
    echo "Downloading and installing uv..."
    curl -LsSf https://astral.sh/uv/install.sh | sh
    
    # Add uv to PATH for current session
    export PATH="$HOME/.cargo/bin:$PATH"
    
    # Check if installation was successful
    if command -v uv &> /dev/null; then
        echo "uv installed successfully: $(uv --version)"
    else
        echo "Warning: uv installation completed but command not found in PATH"
        echo "You may need to restart your shell or run: source ~/.bashrc (or ~/.zshrc)"
    fi
fi

echo ""
echo "================================"
echo "Installation Summary"
echo "================================"
echo ""

# Verify installations
echo "Checking installed packages..."
echo ""

if command -v vim &> /dev/null; then
    echo "✓ vim: $(vim --version | head -n 1)"
else
    echo "✗ vim: NOT FOUND"
fi

if command -v git &> /dev/null; then
    echo "✓ git: $(git --version)"
else
    echo "✗ git: NOT FOUND"
fi

if command -v uv &> /dev/null; then
    echo "✓ uv: $(uv --version)"
else
    echo "✗ uv: NOT FOUND (may need to restart shell)"
fi

echo ""
echo "================================"
echo "Setup complete!"
echo "================================"
echo ""
echo "Note: If uv is not available, you may need to:"
echo "  1. Restart your terminal/shell"
echo "  2. Or run: source ~/.bashrc (or source ~/.zshrc for zsh)"
echo ""
