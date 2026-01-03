# 🛡️ IArgos

## Intelligent Investigation Data Analysis System

**IArgos** is a prototype of an AI-assisted investigation workspace designed to help analysts and investigators **organize, categorize, search, and prioritize large collections of images** based on an investigation context.

The project explores how **multimodal AI (vision + language)** can augment human decision-making in sensitive analytical workflows, rather than replace it.

> ⚠️ This repository is a **research / prototype project**. It is not production-ready and makes explicit design trade-offs for clarity, experimentation, and explainability.

---

## 🎯 Problem Addressed

In many investigative or analytical contexts (law enforcement, journalism, compliance, intelligence, auditing), analysts face:

* Hundreds of heterogeneous images
* Weak or missing metadata
* High cognitive load when deciding *what matters*
* Manual, error-prone triage processes

**IArgos** proposes a structured workflow where AI assists with:

* First-pass categorization
* Context-aware relevance scoring
* Visual exploration and filtering

while keeping the **human fully in control of final judgment**.

---

## 🧠 Core Capabilities

### 1. Context-Aware Analysis

* The user defines an **investigation context** (free-text description).
* All downstream scoring and prioritization explicitly reference this context.

### 2. Image Understanding (Vision + Language)

* Automatic image captioning and visual question answering
* Extraction of descriptive text and semantic tags

### 3. Automatic Categorization

* Each image is assigned to one or more high-level semantic categories (e.g. people, vehicles, documents, weapons, indoor/outdoor, etc.)
* Explicit conflict resolution logic (e.g. indoor vs outdoor)
* Transparent rule-based post-processing on top of model outputs

### 4. Semantic Search

* Textual search over generated descriptions
* Synonym-aware and multilingual-friendly queries

### 5. Investigation Relevance Scoring

* Custom relevance score (0–100) combining:

  * Detected categories
  * Richness of description
  * Semantic overlap with investigation context
  * Exact and partial keyword matches
* Images classified into three operational buckets:

  * 🟢 Relevant
  * 🟡 To review
  * 🔴 Likely irrelevant

### 6. Analyst-Centric UI

* Multi-page Gradio interface
* Clickable statistics and filters
* Progressive disclosure: overview → drill-down

---

## 🏗️ System Design Philosophy

This project intentionally avoids an end-to-end opaque model.

Design choices emphasize:

* **Interpretability over raw accuracy**
* **Deterministic scoring logic layered on top of AI outputs**
* **Explicit heuristics that can be inspected, modified, or debated**

This makes IArgos suitable as:

* A research prototype
* A discussion artifact with domain experts
* A foundation for more robust, audited systems

---

## 🧪 Technical Stack

* **Language**: Python
* **UI**: Gradio (multi-page Blocks)
* **Vision-Language Model**: BLIP (captioning + VQA)
* **Architecture**:

  * Stateful session-based analysis
  * No persistence (in-memory only)
  * Modular scoring and categorization pipeline

---

## 🔬 What This Project Is (and Is Not)

**It is:**

* A serious exploration of AI-assisted analytical workflows
* A demonstration of multimodal reasoning pipelines
* A portfolio project showcasing system-level thinking

**It is not:**

* A production-grade forensic tool
* A claim of automated decision-making
* A benchmark-optimized ML system

---

## 📌 Current Limitations

* Heuristic-based scoring (not learned end-to-end)
* No long-term storage or audit logging
* Performance not optimized for large-scale datasets
* Model bias and error propagation not formally evaluated

These limitations are **explicit and intentional** at this stage.

---

## 🚀 Possible Extensions

* Replace heuristic scoring with learned ranking models
* Add embedding-based semantic similarity (instead of string matching)
* Introduce active learning from analyst feedback
* Formal evaluation on real investigative datasets
* Stronger security, logging, and access controls

---

## 👤 Author Intent

This repository is part of a broader effort to explore:

* Human-centered AI
* AI as a decision-support system
* Practical multimodal pipelines beyond demos

It is shared for **discussion, critique, and learning**, not as a finished product.

---

## ⚖️ License

MIT License

---

If you are a researcher or practitioner interested in investigative AI, explainable systems, or applied multimodal pipelines, this project is meant to be read, questioned, and challenged.
