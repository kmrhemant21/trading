# Practice Activity: Implementing Version Control for Reproducibility

## Activity Instructions

In this activity, you will practice quantifying changes to a piece of pseudo software and assigning appropriate version numbers based on the changes made. Versioning software is crucial for tracking progress, maintaining stability, and managing updates. 

By the end of this activity, you will be able to: 

- Quantify changes made to software projects based on impact.
- Assign appropriate version numbers following semantic versioning principles.

Follow the steps below to complete the activity.

---

## Step-by-Step Guide

### Step 1: Review the Pseudo Software Changes

Below are three sets of changes made to a simple pseudo software project. Review each set of changes carefully.

#### Change Set 1
- Updated the software’s user interface to improve usability.
- Fixed a minor bug that caused the program to crash under specific conditions.
- No changes made to core functionality.

#### Change Set 2
- Added a new feature that allows users to export data in CSV format.
- Refactored the codebase to improve performance and maintainability.
- No breaking changes were introduced.

#### Change Set 3
- A major overhaul of the software’s core functionality changed the way data is processed.
- Introduced backward-incompatible changes, requiring users to update their datasets.
- Added support for a new file format (JSON) for data import.

---

### Step 2: Quantify the Changes

Using the guidelines below, quantify the changes in each set:

- **Patch version (X.X.1):** Minor bug fixes or improvements that do not alter the software’s functionality or API.
- **Minor version (X.1.X):** New features or enhancements that add functionality without breaking existing functionality or APIs.
- **Major version (1.X.X):** Significant changes that include breaking changes, major new features, or overhauls of core functionality.

Assign a version number to each set of changes based on the level of impact.

#### Example for Change Set 1
**Quantified Changes:**
- UI improvement and bug fix.
- No change in core functionality.

**Version Number:** `1.0.1`

**Explanation:**  
This change set includes a minor bug fix and an improvement to the user interface, which qualifies it as a patch update.

---

### Step 3: Assign Version Numbers

Now, assign appropriate version numbers to change sets 2 and 3.

#### Example for Change Set 2
**Quantified Changes:**
- Added a new feature (CSV export) and refactored code.
- No breaking changes.

**Version Number:** `1.1.0`

**Explanation:**  
This change set introduces a new feature and significant code refactoring without breaking existing functionality, qualifying it as a minor version update.

#### Example for Change Set 3
**Quantified Changes:**
- Overhaul of core functionality, backward-incompatible changes, and new file format support.

**Version Number:** `2.0.0`

**Explanation:**  
This change set includes significant changes to the core functionality with backward-incompatible updates, making it a major version update.

---

### Step 4: Reflect on Your Responses

After assigning version numbers to each change set, compare your answers with the explanations provided. Reflect on how you quantified the changes and whether your versioning decisions align with best practices.

---

## Conclusion

In this activity, you practiced the essential skill of quantifying changes in software development and assigning appropriate version numbers based on semantic versioning principles. By reviewing the change sets, you learned to categorize updates as patch, minor, or major versions, which is vital for effective software management. This understanding helps ensure clear communication about the nature of updates, facilitating collaboration and maintaining stability within projects. As you continue working on software projects, applying these versioning principles will enhance your ability to track progress and manage releases efficiently.
