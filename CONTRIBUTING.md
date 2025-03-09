# Contributing to Bregma RL Portfolio

Thank you for your interest in contributing to this portfolio optimization project! Here's how you can help.

## Development Setup

1. Fork the repository
2. Clone your fork locally:
   ```bash
   git clone https://github.com/your-username/bregma-rl-portfolio.git
   cd bregma-rl-portfolio
   ```

3. Choose your platform-specific setup method:

### Linux
```bash
chmod +x setup.sh
./setup.sh
```

### Windows
```cmd
setup_windows.bat
```

### macOS (Intel)
```bash
chmod +x setup.sh
./setup.sh
```

### macOS (Apple Silicon)
```bash
chmod +x setup_arm64.sh
./setup_arm64.sh
```

### Docker
```bash
# CPU-only
docker-compose up bregma-rl-cpu

# Or with GPU support
docker-compose up bregma-rl
```

## Contribution Guidelines

### Code Style
- Follow PEP 8 guidelines for Python code
- Use type hints where appropriate
- Write docstrings for functions and classes

### Pull Request Process
1. Create a new branch for your feature or bugfix
2. Make your changes
3. Ensure all tests pass
4. Update documentation as needed
5. Submit a pull request with a clear description of the changes

### Adding New Features
When adding new features, please:
- Include appropriate tests
- Update the README if necessary
- Document the feature's functionality

### Reporting Issues
When reporting issues, please include:
- A clear description of the issue
- Steps to reproduce the issue
- Expected and actual behavior
- Environment information (OS, Python version, etc.)

## Feature Roadmap

Here are some areas where contributions would be particularly valuable:

1. Additional RL algorithms (PPO, SAC, etc.)
2. Enhanced feature engineering techniques
3. Improved market regime detection
4. Transaction cost modeling
5. Portfolio risk constraints
6. Hyperparameter tuning and optimization

## License

By contributing to this project, you agree that your contributions will be licensed under the project's license (MIT License).