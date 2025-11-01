# Practice activity: Designing an intelligent troubleshooting agent

**Disclaimer**: You will create a troubleshooting agent in this activity. If you are currently taking the course "Building Intelligent Troubleshooting Agents," proceed with this activity.

If you are taking the course "Microsoft Azure for AI and Machine Learning" and already took the course "Building Intelligent Troubleshooting Agents," you can skip this activity and use that agent. Otherwise, you must complete this activity in order to move forward.

## Introduction

In this activity, you will design a basic intelligent troubleshooting agent that can diagnose and resolve simple technical issues. This activity will guide you through the process of setting up the environment, defining the agent's capabilities, and creating the logic that allows it to interact with users and provide solutions.

By the end of this activity, you will be able to:

- List the key components required to design an intelligent troubleshooting agent.
- Implement basic functionality for user interaction, problem diagnosis, and solution suggestion.
- Integrate a knowledge base that the agent can use to provide troubleshooting advice.

## Step-by-step process for designing an intelligent troubleshooting agent

Create a new Jupyter notebook. You can call it "troubleshooting_agent". Make sure you have the appropriate Python kernel selected. Also create a file for your knowledge base. You can call it "troubleshooting_knowledge_base.json". This setup provides us with the tools to process user input and interaction with the troubleshooting agent.

The remaining of this reading will guide you through the following steps:

1. Step 1: Setting up the environment
2. Step 2: Define the knowledge base
3. Step 3: Implement user interaction
4. Step 4: Add diagnostic logic
5. Step 5: Automate common fixes
6. Step 6: Test and improve the agent
7. Step 7: Reflection and make further enhancements

### Step 1: Set up the environment

**Instructions**

You will need a programming environment capable of handling natural language processing (NLP), decision-making logic, and basic data retrieval. Python is recommended for this task, along with libraries like NLTK, spaCy, or Transformers for natural language understanding.

Ensure you have access to a simple knowledge base. This can be a static file (such as a .json or .csv file) that contains predefined troubleshooting steps and solutions for common problems. You will create a sample knowledge base or utilize the sample knowledge base code provided below, which contains its own knowledge base.

**Code example - Sample Knowledge base**

```python
# Sample Knowledge Base for Network Troubleshooting
knowledge_base = {
"restart_router": "Please restart your router and check if the problem persists.",
"reset_network_settings": "Try resetting your network settings. Instructions are available in your system settings under 'Network Reset'.",
"check_cables": "Ensure all network cables are securely connected.",
"isp_contact": "If the issue continues, please contact your Internet Service Provider (ISP) for further assistance.",
"clear_cache": "Clearing your browser cache can sometimes resolve connectivity issues."
}

def diagnose_network_issue():
print("Let's diagnose your network issue.")
```

**Code example**

```python
response = input("Have you tried restarting your router? (Yes/No): ").strip().lower()
if response == "no":
print(knowledge_base["restart_router"])
return # Exit after suggesting a solution
```

### Step 2: Check if the cables are connected properly

**Instructions**

Create a basic knowledge base that stores common issues and their corresponding solutions. You can start with a small list of five to ten problems and solutions. The agent will search this knowledge base to match user-reported problems with known issues.

**Example knowledge base (JSON format)**

```python
response = input("Are all cables securely connected? (Yes/No): ").strip().lower()
if response == "no":
print(knowledge_base["check_cables"])
return # Exit after suggesting a solution
```

Save this knowledge base as troubleshooting_knowledge_base.json

### Step 3: Check if network settings need resetting

**Instructions**

Implement a simple user interface that allows the user to describe their problem. You can start by asking the user to input their issue in natural language.

Use NLP to interpret the user's input and extract the key symptoms that match issues in the knowledge base.

**Code example**

```python
response = input("Would you like to try resetting your network settings? (Yes/No): ").strip().lower()
if response == "yes":
    print(knowledge_base["reset_network_settings"])
    return # Exit after suggesting a solution
```

### Step 4: Suggest clearing browser cache as a possible solution

**Instructions**

Extend the agent's logic by adding more sophisticated diagnostic capabilities. You can implement decision trees or rule-based systems to guide the agent through a sequence of troubleshooting steps based on user responses.

For example, if the issue is related to network problems, the agent can ask follow-up questions, such as whether the user has restarted their router or checked their internet settings.

**Code example**

```python
response = input("Is this issue occurring in your browser? (Yes/No): ").strip().lower()
if response == "yes":
    print(knowledge_base["clear_cache"])
else:
    print(knowledge_base["isp_contact"])

# Example of triggering diagnostic logic based on user input
user_input = input("Please describe your issue: ").strip().lower()
if "network" in user_input:
    diagnose_network_issue()
```

### Step 5: Automate common fixes

**Instructions**

Add functionality to automate certain troubleshooting steps for common problems. For example, the agent could suggest or automatically reset settings if a common issue is detected (this can be simulated in the activity).

**Code example**

```python
def automate_fix(issue):
    if issue == "network_issue":
        print("Attempting to reset your network settings automatically...")
        # Simulate network reset
        print("Network settings have been reset. Please check your connection.")
    else:
        print("Automatic fix is not available for this issue.")

# Simulate automated fix
if "network" in user_input.lower():
    automate_fix("network_issue")
```

### Step 6: Test and improve the agent

**Instructions**

Test your troubleshooting agent by entering a variety of problems and observing its responses. Pay attention to whether the agent can correctly identify the problem and offer a solution.

Improve the agent by adding more diagnostic pathways, expanding the knowledge base, and refining the user interaction experience.

**Example test cases**

**Case 1**

Input: "My internet is very slow."

Expected output: "Restart your router and check your network settings."

**Case 2**

Input: "The software keeps crashing on my computer."

Expected output: "Update the software to the latest version and restart your computer."

### Step 7: Reflect and make further enhancements

After completing the activity, reflect on the following questions:

- What improvements can you make to the agent's diagnostic capabilities?
- How can you make the user interaction more natural and conversational?
- How would you handle more complex troubleshooting tasks that require deeper diagnostic logic?

**Further enhancements**

- Consider integrating ML models to allow the agent to learn from past interactions and improve its troubleshooting recommendations over time.
- Expand the knowledge base to cover a broader range of issues and add more complex solutions.

## Conclusion

Through this activity, you have implemented a basic intelligent troubleshooting agent capable of interacting with users, diagnosing issues, and offering solutions. While this agent handles simple cases, the foundational principles learned here can be expanded to create more sophisticated agents capable of solving complex problems in real-world scenarios.