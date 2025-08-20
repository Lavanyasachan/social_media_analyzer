"""
Fine-tuning Script for DSA Coding Agent using PEFT and Unsloth

This script demonstrates how the Llama 3.1 model was fine-tuned specifically
for DSA problem solving using Parameter-Efficient Fine-Tuning (PEFT) with
LoRA (Low-Rank Adaptation) and Unsloth for optimized training.

The fine-tuned model is then deployed to Ollama for direct use in the multi-agent system.
"""

import torch
from transformers import AutoTokenizer, AutoModelForCausalLM, TrainingArguments, Trainer
from peft import LoraConfig, get_peft_model, TaskType, prepare_model_for_kbit_training
from unsloth import FastLanguageModel
import datasets
import json
import os
from typing import List, Dict
import subprocess
from config import PEFT_CONFIG, UNSLOTH_CONFIG


class DSAFineTuner:
    """
    Fine-tuning system for creating specialized DSA coding agent
    """
    
    def __init__(self, model_name: str = "unsloth/llama-3.1-8b-bnb-4bit"):
        self.model_name = model_name
        self.max_seq_length = UNSLOTH_CONFIG["max_seq_length"]
        self.dtype = UNSLOTH_CONFIG["dtype"]
        self.load_in_4bit = UNSLOTH_CONFIG["load_in_4bit"]
        
        # Initialize model and tokenizer
        self.model = None
        self.tokenizer = None
        self.setup_model()
    
    def setup_model(self):
        """Initialize the base model with Unsloth optimizations"""
        print("Loading base model with Unsloth optimizations...")
        
        self.model, self.tokenizer = FastLanguageModel.from_pretrained(
            model_name=self.model_name,
            max_seq_length=self.max_seq_length,
            dtype=self.dtype,
            load_in_4bit=self.load_in_4bit
        )
        
        # Configure PEFT with LoRA
        self.model = FastLanguageModel.get_peft_model(
            self.model,
            r=PEFT_CONFIG["rank"],
            target_modules=PEFT_CONFIG["target_modules"],
            lora_alpha=PEFT_CONFIG["alpha"],
            lora_dropout=PEFT_CONFIG["dropout"],
            bias="none",
            use_gradient_checkpointing=UNSLOTH_CONFIG["use_gradient_checkpointing"],
            random_state=3407,
            use_rslora=False,
            loftq_config=None,
        )
        
        print("Model setup complete with PEFT configuration")
    
    def create_dsa_training_dataset(self) -> List[Dict]:
        """
        Create comprehensive DSA training dataset with problem-solution pairs
        """
        training_data = [
            {
                "instruction": "Solve the Two Sum problem",
                "input": "Given an array of integers nums and an integer target, return indices of the two numbers such that they add up to target. Example: nums = [2,7,11,15], target = 9, output = [0,1]",
                "output": """def twoSum(nums, target):
    # Use hash map for O(n) solution
    num_map = {}
    
    for i, num in enumerate(nums):
        complement = target - num
        if complement in num_map:
            return [num_map[complement], i]
        num_map[num] = i
    
    return []  # No solution found

# Time Complexity: O(n)
# Space Complexity: O(n)"""
            },
            {
                "instruction": "Implement Binary Tree Inorder Traversal",
                "input": "Given the root of a binary tree, return the inorder traversal of its nodes' values.",
                "output": """def inorderTraversal(root):
    # Recursive approach
    def inorder(node):
        if not node:
            return []
        return inorder(node.left) + [node.val] + inorder(node.right)
    
    return inorder(root)

# Alternative iterative approach:
def inorderTraversalIterative(root):
    result = []
    stack = []
    current = root
    
    while stack or current:
        # Go to leftmost node
        while current:
            stack.append(current)
            current = current.left
        
        # Process current node
        current = stack.pop()
        result.append(current.val)
        
        # Move to right subtree
        current = current.right
    
    return result

# Time Complexity: O(n)
# Space Complexity: O(h) where h is height of tree"""
            },
            {
                "instruction": "Solve Longest Common Subsequence using Dynamic Programming",
                "input": "Given two strings text1 and text2, return the length of their longest common subsequence. If there is no common subsequence, return 0.",
                "output": """def longestCommonSubsequence(text1, text2):
    m, n = len(text1), len(text2)
    
    # Create DP table
    dp = [[0] * (n + 1) for _ in range(m + 1)]
    
    # Fill the DP table
    for i in range(1, m + 1):
        for j in range(1, n + 1):
            if text1[i-1] == text2[j-1]:
                dp[i][j] = dp[i-1][j-1] + 1
            else:
                dp[i][j] = max(dp[i-1][j], dp[i][j-1])
    
    return dp[m][n]

# Optimized space complexity version:
def longestCommonSubsequenceOptimized(text1, text2):
    m, n = len(text1), len(text2)
    
    # Use only two rows instead of full matrix
    prev = [0] * (n + 1)
    curr = [0] * (n + 1)
    
    for i in range(1, m + 1):
        for j in range(1, n + 1):
            if text1[i-1] == text2[j-1]:
                curr[j] = prev[j-1] + 1
            else:
                curr[j] = max(prev[j], curr[j-1])
        prev, curr = curr, prev
    
    return prev[n]

# Time Complexity: O(m*n)
# Space Complexity: O(min(m,n)) for optimized version"""
            },
            {
                "instruction": "Implement Quick Sort algorithm",
                "input": "Write a function to sort an array using the quicksort algorithm with proper partitioning.",
                "output": """def quickSort(arr):
    def partition(low, high):
        # Choose rightmost element as pivot
        pivot = arr[high]
        i = low - 1  # Index of smaller element
        
        for j in range(low, high):
            # If current element is smaller than or equal to pivot
            if arr[j] <= pivot:
                i += 1
                arr[i], arr[j] = arr[j], arr[i]
        
        # Place pivot in correct position
        arr[i + 1], arr[high] = arr[high], arr[i + 1]
        return i + 1
    
    def quickSortHelper(low, high):
        if low < high:
            # Partition index
            pi = partition(low, high)
            
            # Recursively sort elements before and after partition
            quickSortHelper(low, pi - 1)
            quickSortHelper(pi + 1, high)
    
    if arr:
        quickSortHelper(0, len(arr) - 1)
    return arr

# Time Complexity: Average O(n log n), Worst O(n²)
# Space Complexity: O(log n) for recursion stack"""
            },
            {
                "instruction": "Solve Maximum Subarray Sum (Kadane's Algorithm)",
                "input": "Given an integer array nums, find the contiguous subarray with the largest sum and return its sum.",
                "output": """def maxSubArray(nums):
    if not nums:
        return 0
    
    # Kadane's Algorithm
    max_sum = float('-inf')
    current_sum = 0
    
    for num in nums:
        current_sum = max(num, current_sum + num)
        max_sum = max(max_sum, current_sum)
    
    return max_sum

# Alternative approach with tracking indices:
def maxSubArrayWithIndices(nums):
    if not nums:
        return 0, -1, -1
    
    max_sum = float('-inf')
    current_sum = 0
    start = 0
    end = 0
    temp_start = 0
    
    for i in range(len(nums)):
        current_sum += nums[i]
        
        if current_sum > max_sum:
            max_sum = current_sum
            start = temp_start
            end = i
        
        if current_sum < 0:
            current_sum = 0
            temp_start = i + 1
    
    return max_sum, start, end

# Time Complexity: O(n)
# Space Complexity: O(1)"""
            },
            {
                "instruction": "Implement Graph DFS and BFS traversal",
                "input": "Write functions to traverse a graph using both Depth-First Search and Breadth-First Search.",
                "output": """from collections import deque, defaultdict

class Graph:
    def __init__(self):
        self.graph = defaultdict(list)
    
    def addEdge(self, u, v):
        self.graph[u].append(v)
    
    def DFS(self, start):
        visited = set()
        result = []
        
        def dfsHelper(vertex):
            visited.add(vertex)
            result.append(vertex)
            
            for neighbor in self.graph[vertex]:
                if neighbor not in visited:
                    dfsHelper(neighbor)
        
        dfsHelper(start)
        return result
    
    def DFSIterative(self, start):
        visited = set()
        stack = [start]
        result = []
        
        while stack:
            vertex = stack.pop()
            if vertex not in visited:
                visited.add(vertex)
                result.append(vertex)
                
                # Add neighbors in reverse order for correct traversal
                for neighbor in reversed(self.graph[vertex]):
                    if neighbor not in visited:
                        stack.append(neighbor)
        
        return result
    
    def BFS(self, start):
        visited = set()
        queue = deque([start])
        result = []
        
        while queue:
            vertex = queue.popleft()
            if vertex not in visited:
                visited.add(vertex)
                result.append(vertex)
                
                for neighbor in self.graph[vertex]:
                    if neighbor not in visited:
                        queue.append(neighbor)
        
        return result

# Time Complexity: O(V + E) where V is vertices and E is edges
# Space Complexity: O(V) for visited set and queue/stack"""
            }
        ]
        
        return training_data
    
    def format_training_data(self, data: List[Dict]) -> datasets.Dataset:
        """Format training data for instruction tuning"""
        formatted_data = []
        
        for item in data:
            # Create instruction-following format
            text = f"""### Instruction:
{item['instruction']}

### Input:
{item['input']}

### Response:
{item['output']}"""
            
            formatted_data.append({"text": text})
        
        return datasets.Dataset.from_list(formatted_data)
    
    def fine_tune_model(self):
        """Execute the fine-tuning process"""
        print("Preparing training dataset...")
        training_data = self.create_dsa_training_dataset()
        dataset = self.format_training_data(training_data)
        
        print("Starting fine-tuning with Unsloth...")
        
        # Training arguments
        training_args = TrainingArguments(
            per_device_train_batch_size=UNSLOTH_CONFIG["per_device_train_batch_size"],
            gradient_accumulation_steps=UNSLOTH_CONFIG["gradient_accumulation_steps"],
            warmup_steps=UNSLOTH_CONFIG["warmup_steps"],
            num_train_epochs=UNSLOTH_CONFIG["num_train_epochs"],
            learning_rate=UNSLOTH_CONFIG["learning_rate"],
            fp16=True if UNSLOTH_CONFIG["dtype"] == "float16" else False,
            logging_steps=UNSLOTH_CONFIG["logging_steps"],
            output_dir="./dsa_finetuned_model",
            save_steps=UNSLOTH_CONFIG["save_steps"],
            save_total_limit=3,
            dataloader_num_workers=0,
            optim="adamw_8bit",
            weight_decay=0.01,
            lr_scheduler_type="linear",
            seed=42,
        )
        
        # Use FastLanguageModel's SFTTrainer
        trainer = FastLanguageModel.get_trainer(
            model=self.model,
            tokenizer=self.tokenizer,
            train_dataset=dataset,
            dataset_text_field="text",
            max_seq_length=self.max_seq_length,
            args=training_args,
        )
        
        # Start training
        print("Training started...")
        trainer.train()
        
        print("Fine-tuning complete!")
        return trainer
    
    def save_model(self, output_dir: str = "./dsa_finetuned_model"):
        """Save the fine-tuned model"""
        print(f"Saving model to {output_dir}")
        self.model.save_pretrained(output_dir)
        self.tokenizer.save_pretrained(output_dir)
        
        # Save PEFT adapter separately
        if hasattr(self.model, 'peft_config'):
            self.model.save_pretrained(f"{output_dir}/peft_adapter")
    
    def export_to_ollama(self, model_name: str = "llama3.1-dsa-coder"):
        """Export the fine-tuned model to Ollama format"""
        print("Exporting model to Ollama...")
        
        # Create Ollama Modelfile
        modelfile_content = f"""FROM ./dsa_finetuned_model

TEMPLATE \"\"\"### Instruction:
{{{{ .Instruction }}}}

### Input:
{{{{ .Input }}}}

### Response:
{{{{ .Response }}}}
\"\"\"

PARAMETER temperature 0.3
PARAMETER top_p 0.9
PARAMETER stop "### Instruction:"
PARAMETER stop "### Input:"
PARAMETER stop "### Response:"

SYSTEM \"\"\"You are a specialized DSA (Data Structures and Algorithms) coding assistant. You provide efficient, well-commented Python solutions for algorithmic problems. Always include time and space complexity analysis.\"\"\"
"""
        
        with open("Modelfile", "w") as f:
            f.write(modelfile_content)
        
        try:
            # Create Ollama model
            result = subprocess.run([
                "ollama", "create", model_name, "-f", "Modelfile"
            ], capture_output=True, text=True)
            
            if result.returncode == 0:
                print(f"Successfully created Ollama model: {model_name}")
            else:
                print(f"Error creating Ollama model: {result.stderr}")
        
        except FileNotFoundError:
            print("Ollama not found. Please install Ollama first.")
            print("You can manually create the model using the generated Modelfile")


def main():
    """Main fine-tuning pipeline"""
    print("DSA Fine-tuning Pipeline using PEFT and Unsloth")
    print("=" * 50)
    
    # Initialize fine-tuner
    fine_tuner = DSAFineTuner()
    
    # Execute fine-tuning
    trainer = fine_tuner.fine_tune_model()
    
    # Save the model
    fine_tuner.save_model()
    
    # Export to Ollama
    fine_tuner.export_to_ollama()
    
    print("\nFine-tuning pipeline complete!")
    print("The model is now ready for use in the multi-agent DSA system.")


if __name__ == "__main__":
    main()