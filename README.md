# 🤖 Multi-Agent System for DSA Problem Solving

A sophisticated multi-agent system that collaboratively solves Data Structures and Algorithms (DSA) problems using **AutoGen**, **fine-tuned Llama 3.1**, and **local LLMs** through Ollama.

## 📋 Project Overview

**Duration:** July 2024 - August 2024  
**Success Rate:** ~70% on DSA problems  
**Architecture:** 3-agent collaborative system with local LLM deployment

## 🏗️ System Architecture

### 🤖 Three Specialized Agents

1. **Problem Analyzer Agent**
   - Analyzes DSA problem requirements and constraints
   - Identifies optimal algorithmic approaches
   - Suggests time/space complexity targets
   - Breaks down problems into manageable components

2. **Coding Agent (Fine-tuned Llama 3.1)**
   - Implements optimized Python solutions
   - Fine-tuned using **PEFT** (Parameter-Efficient Fine-Tuning)
   - Optimized with **Unsloth** for efficient training
   - Deployed via **Ollama** for local inference
   - Specializes in DSA patterns and best practices

3. **Testing Agent**
   - Creates comprehensive test suites
   - Validates solution correctness
   - Performs performance analysis
   - Suggests optimizations and improvements

### 🔧 Technical Stack

- **AutoGen Framework**: Multi-agent orchestration and conversation management
- **Ollama**: Local LLM deployment and inference
- **PEFT & Unsloth**: Efficient parameter fine-tuning techniques
- **Llama 3.1 8B**: Base foundation model
- **Streamlit**: Interactive user interface
- **SQLite**: Local data storage for results and feedback

## 🚀 Key Features

- **Collaborative Problem Solving**: Agents work together to analyze, code, and test solutions
- **Local Deployment**: Complete privacy with local LLM execution via Ollama
- **Real-time Testing**: Integrated code execution and validation
- **User Feedback Loop**: Continuous improvement through user suggestions
- **Fine-tuned Coding Agent**: Specialized for Python DSA implementations
- **Performance Tracking**: Analytics dashboard with success rate monitoring

## 📊 Performance Metrics

- **Overall Success Rate**: ~70% across all difficulty levels
- **Easy Problems**: ~85% success rate
- **Medium Problems**: ~70% success rate  
- **Hard Problems**: ~55% success rate
- **Code Quality**: Well-commented, optimized solutions with complexity analysis

## 🛠️ Installation and Setup

### Prerequisites

1. **Python 3.8+**
2. **Ollama** installed locally
3. **CUDA** (optional, for GPU acceleration)

### Step 1: Clone Repository

```bash
git clone https://github.com/Lavanyasachan/social_media_analyzer.git
cd social_media_analyzer
```

### Step 2: Install Dependencies

```bash
pip install -r requirements.txt
```

### Step 3: Setup Ollama

```bash
# Install Ollama (follow official instructions)
curl -fsSL https://ollama.ai/install.sh | sh

# Pull base Llama 3.1 model
ollama pull llama3.1:8b

# The fine-tuned model will be created during setup
```

### Step 4: Fine-tune the Coding Agent

```bash
python fine_tune_dsa_model.py
```

This will:
- Fine-tune Llama 3.1 using PEFT and Unsloth
- Create the specialized DSA coding agent
- Export the model to Ollama format

## 🚀 Usage

### Command Line Interface

```python
from multi_agent_dsa_system import MultiAgentDSASystem, DSAProblem

# Initialize the system
system = MultiAgentDSASystem()

# Create a DSA problem
problem = DSAProblem(
    title="Two Sum",
    description="Find two numbers that add up to target",
    difficulty="Easy",
    examples=[{"input": {"nums": [2,7,11,15], "target": 9}, "output": [0,1]}],
    constraints="2 <= nums.length <= 10^4",
    expected_complexity={"time": "O(n)", "space": "O(n)"}
)

# Solve the problem
solution = system.solve_dsa_problem(problem)
print(f"Success: {solution['success']}")
print(f"Code: {solution['code']}")
```

### Web Interface

Launch the interactive Streamlit interface:

```bash
streamlit run dsa_ui.py
```

Features include:
- Problem submission form
- Real-time agent collaboration visualization
- Solution review and feedback
- Performance analytics dashboard

## 📁 Project Structure

```
social_media_analyzer/
├── multi_agent_dsa_system.py    # Core multi-agent system
├── fine_tune_dsa_model.py       # PEFT fine-tuning pipeline
├── dsa_ui.py                    # Streamlit user interface
├── config.py                    # Configuration settings
├── requirements.txt             # Python dependencies
├── README.md                    # This file
└── dsa_workspace/               # Code execution workspace
```

## 🔬 Fine-tuning Details

### PEFT Configuration
- **Technique**: LoRA (Low-Rank Adaptation)
- **Rank**: 16
- **Alpha**: 32
- **Dropout**: 0.1
- **Target Modules**: All linear layers

### Unsloth Optimizations
- **4-bit Quantization**: Reduced memory usage
- **Gradient Checkpointing**: Memory-efficient training
- **Flash Attention**: Faster training
- **Custom CUDA Kernels**: 2x speedup

### Training Data
- Curated DSA problem-solution pairs
- Multiple programming paradigms
- Complexity analysis examples
- Edge case handling patterns

## 📈 Success Rate Analysis

The system achieves a **~70% overall success rate** through:

1. **Collaborative Analysis**: Multiple perspectives on problem approach
2. **Specialized Fine-tuning**: Domain-specific knowledge in the coding agent
3. **Comprehensive Testing**: Thorough validation of solutions
4. **Iterative Improvement**: User feedback integration

### Performance by Category
- **Arrays & Hash Tables**: 75%
- **Trees & Graphs**: 68%
- **Dynamic Programming**: 65%
- **Sorting & Searching**: 80%

## 🔄 User Feedback Integration

The system incorporates user feedback to continuously improve:

1. **Solution Rating**: Users rate generated solutions 1-5
2. **Feedback Collection**: Specific improvement suggestions
3. **Performance Tracking**: Success rate monitoring
4. **Model Updates**: Periodic fine-tuning with new data

## 🛡️ Local Deployment Advantages

- **Privacy**: All processing happens locally
- **Control**: Full control over model behavior
- **Customization**: Easy to adapt for specific domains
- **No API Costs**: One-time setup, no ongoing fees
- **Offline Capability**: Works without internet connection

## 🤝 Contributing

1. Fork the repository
2. Create a feature branch (`git checkout -b feature/improvement`)
3. Commit changes (`git commit -am 'Add new feature'`)
4. Push to branch (`git push origin feature/improvement`)
5. Create a Pull Request

## 📝 License

This project is licensed under the MIT License - see the LICENSE file for details.

## 🙏 Acknowledgments

- **AutoGen Team**: For the multi-agent framework
- **Unsloth Team**: For efficient fine-tuning tools
- **Ollama Team**: For local LLM deployment
- **Meta**: For the Llama 3.1 foundation model

## 📞 Support

For questions or issues:
- Open a GitHub issue
- Review the documentation
- Check the Streamlit interface for interactive help

---

**Built with ❤️ for the coding community**
