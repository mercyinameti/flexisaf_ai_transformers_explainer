"""
Illustrated Transformer Explainer
Author: Mercy Inameti
FlexiStaff AI Engineering - Week I2

Demonstrates transformer concepts without heavy dependencies
"""

import json
import numpy as np
import matplotlib.pyplot as plt
from sklearn.decomposition import PCA
import warnings
warnings.filterwarnings('ignore')

print("\n" + "="*30)
print("TRANSFORMER EXPLAINER")
print("="*30)

# ============================================
# SECTION 1: TOKENIZATION EXPLANATION
# ============================================

print("\n" + "="*60)
print("SECTION 1: TOKENIZATION DEMONSTRATION")
print("="*60)

texts = [
    "Hello world",
    "The quick brown fox jumps",
    "I don't understand transformers",
    "Running machine learning models",
    "Artificial intelligence is transforming technology"
]

tokenization_results = []

for text in texts:
    words = text.lower().split()
    tokens = []
    for word in words:
        if word == "don't":
            tokens.extend(["do", "n't"])
        elif word == "understanding":
            tokens.extend(["understand", "ing"])
        else:
            tokens.append(word)
    
    result = {
        "text": text,
        "tokens": tokens,
        "token_count": len(tokens),
        "character_count": len(text)
    }
    tokenization_results.append(result)
    
    print(f"\nText: '{text}'")
    print(f"  Tokens: {tokens}")
    print(f"  Token count: {len(tokens)}")

with open("tokenization_results.json", "w", encoding='utf-8') as f:
    json.dump(tokenization_results, f, indent=2)

print("\nCheckmark: Tokenization results saved")

# ============================================
# SECTION 2: TOKEN-COUNT COMPARISON
# ============================================

print("\n" + "="*60)
print("SECTION 2: TOKEN-COUNT COMPARISON")
print("="*60)

comparison_texts = [
    ("Simple word", "cat"),
    ("Contraction", "don't"),
    ("Phrase", "The feline sat on the rug"),
    ("Longer phrase", "A small domesticated furry animal was positioned upon a floor covering"),
]

print("\nComparing token counts:\n")
comparison_results = []

for label, text in comparison_texts:
    words = text.lower().split()
    token_count = len(words) + 2
    
    comparison_results.append({
        "label": label,
        "text": text,
        "word_count": len(words),
        "estimated_tokens": token_count
    })
    print(f"{label:25} | Words: {len(words):2} | Est. Tokens: {token_count:2}")

# ============================================
# SECTION 3: EMBEDDINGS & SIMILARITY
# ============================================

print("\n" + "="*60)
print("SECTION 3: EMBEDDINGS AND SIMILARITY")
print("="*60)

embedding_texts = [
    "The cat sat on the mat",
    "A feline rested on the rug",
    "The dog ran in the park",
    "A canine sprinted through the garden",
    "The weather is sunny today"
]

manual_embeddings = [
    [0.9, 0.2, 0.1, 0.0, 0.0],
    [0.85, 0.25, 0.1, 0.0, 0.0],
    [0.1, 0.2, 0.9, 0.0, 0.0],
    [0.05, 0.25, 0.95, 0.0, 0.0],
    [0.2, 0.2, 0.2, 0.9, 0.0]
]

embeddings = []
for i, text in enumerate(embedding_texts):
    embeddings.append({
        "text": text,
        "embedding": manual_embeddings[i]
    })
    print(f"  Checkmark Embedded: '{text}'")

print("\nCalculating cosine similarity:\n")

def cosine_similarity(vec1, vec2):
    vec1 = np.array(vec1)
    vec2 = np.array(vec2)
    return np.dot(vec1, vec2) / (np.linalg.norm(vec1) * np.linalg.norm(vec2))

similarities = []
for i in range(len(embeddings)):
    for j in range(i+1, len(embeddings)):
        sim = cosine_similarity(
            embeddings[i]["embedding"],
            embeddings[j]["embedding"]
        )
        similarities.append({
            "text1": embeddings[i]["text"][:30],
            "text2": embeddings[j]["text"][:30],
            "similarity": float(sim)
        })
        if sim > 0.7:
            print(f"  SIMILAR: '{embeddings[i]['text'][:25]}' <-> '{embeddings[j]['text'][:25]}' = {sim:.3f}")
        elif sim > 0.4:
            print(f"  MODERATE: '{embeddings[i]['text'][:25]}' <-> '{embeddings[j]['text'][:25]}' = {sim:.3f}")

with open("embedding_similarities.json", "w", encoding='utf-8') as f:
    json.dump(similarities, f, indent=2)

# ============================================
# SECTION 4: EMBEDDING VISUALIZATION
# ============================================

print("\n" + "="*60)
print("SECTION 4: EMBEDDING VISUALIZATION")
print("="*60)

embeddings_array = np.array([e["embedding"] for e in embeddings])
pca = PCA(n_components=2)
embeddings_2d = pca.fit_transform(embeddings_array)

plt.figure(figsize=(10, 8))
colors = ['red', 'red', 'blue', 'blue', 'green']

for i, (x, y) in enumerate(embeddings_2d):
    plt.scatter(x, y, s=200, c=colors[i], alpha=0.6, edgecolors='black', linewidth=2)
    plt.annotate(f"{i+1}", (x, y), fontsize=12, fontweight='bold', ha='center', va='center')

plt.xlabel("PCA Dimension 1", fontsize=12)
plt.ylabel("PCA Dimension 2", fontsize=12)
plt.title("Embedding Space Visualization\n(Similar sentences cluster together)", fontsize=14, fontweight='bold')
plt.grid(True, alpha=0.3)

from matplotlib.patches import Patch
legend_elements = [
    Patch(facecolor='red', alpha=0.6, edgecolor='black', label='Cat sentences (similar)'),
    Patch(facecolor='blue', alpha=0.6, edgecolor='black', label='Dog sentences (similar)'),
    Patch(facecolor='green', alpha=0.6, edgecolor='black', label='Weather (different)')
]
plt.legend(handles=legend_elements, loc='best')

plt.tight_layout()
plt.savefig("embedding_visualization.png", dpi=150, bbox_inches='tight')
print("\nCheckmark: Embedding visualization saved")
plt.close()

# ============================================
# SECTION 4B: TRANSFORMER ARCHITECTURE
# ============================================

print("\n" + "="*60)
print("SECTION 4B: TRANSFORMER ARCHITECTURE DIAGRAM")
print("="*60)

fig, ax = plt.subplots(figsize=(14, 10))
ax.set_xlim(0, 10)
ax.set_ylim(0, 12)
ax.axis('off')

ax.text(5, 11.5, 'Transformer Architecture (Attention Is All You Need)', 
        fontsize=16, fontweight='bold', ha='center')

def draw_box(ax, x, y, width, height, text, color='lightblue'):
    rect = plt.Rectangle((x-width/2, y-height/2), width, height, 
                         edgecolor='black', facecolor=color, linewidth=2)
    ax.add_patch(rect)
    ax.text(x, y, text, fontsize=9, ha='center', va='center', fontweight='bold')

draw_box(ax, 5, 10.5, 1.5, 0.6, 'Input\n(Tokens)', 'lightgreen')
ax.arrow(5, 10.2, 0, -0.4, head_width=0.15, head_length=0.1, fc='black', ec='black')

draw_box(ax, 5, 9.5, 1.8, 0.6, 'Token\nEmbedding', 'lightyellow')
ax.arrow(5, 9.2, 0, -0.4, head_width=0.15, head_length=0.1, fc='black', ec='black')

draw_box(ax, 2, 8.5, 1.8, 0.6, 'Positional\nEncoding', 'lightcyan')
ax.arrow(2, 8.2, 2.3, -0.5, head_width=0.15, head_length=0.1, fc='black', ec='black')

draw_box(ax, 5, 7.8, 0.6, 0.6, '+', 'white')
ax.arrow(5, 7.5, 0, -0.4, head_width=0.15, head_length=0.1, fc='black', ec='black')

for i in range(3):
    block_y = 6.5 - (i * 1.5)
    draw_box(ax, 1.5, block_y, 2, 0.7, 'Multi-Head\nAttention', 'lightcoral')
    ax.arrow(5, block_y + 0.5, -2.5, -0.2, head_width=0.15, head_length=0.1, fc='black', ec='black')
    
    draw_box(ax, 3.2, block_y - 0.8, 1.5, 0.6, 'Add and Norm', 'lightblue')
    ax.arrow(1.5, block_y - 0.5, 1.2, -0.5, head_width=0.12, head_length=0.08, fc='black', ec='black')
    
    draw_box(ax, 5.5, block_y, 2, 0.7, 'Feed Forward\n(MLP)', 'lightsalmon')
    ax.arrow(3.2, block_y - 1.3, 1.8, 0.3, head_width=0.12, head_length=0.08, fc='black', ec='black')
    
    draw_box(ax, 6.8, block_y - 0.8, 1.5, 0.6, 'Add and Norm', 'lightblue')
    ax.arrow(5.5, block_y - 0.5, 1.0, -0.5, head_width=0.12, head_length=0.08, fc='black', ec='black')
    
    if i < 2:
        ax.arrow(6.8, block_y - 1.3, -1.5, -0.7, head_width=0.12, head_length=0.08, fc='black', ec='black')

draw_box(ax, 5, 1.5, 2, 0.6, 'Output Layer', 'lightgoldenrodyellow')
ax.arrow(6.8, 2.5 - 0.5, -0.9, -0.6, head_width=0.12, head_length=0.08, fc='black', ec='black')

draw_box(ax, 5, 0.8, 2, 0.6, 'Softmax and Argmax', 'lightsteelblue')
ax.arrow(5, 1.2, 0, -0.4, head_width=0.15, head_length=0.1, fc='black', ec='black')

draw_box(ax, 5, 0.1, 1.5, 0.5, 'Predicted Token', 'lightgreen')

plt.tight_layout()
plt.savefig("transformer_architecture_diagram.png", dpi=150, bbox_inches='tight')
print("Checkmark: Transformer architecture diagram saved")
plt.close()

# ============================================
# SECTION 5: PROMPTING VS FINE-TUNING
# ============================================

print("\n" + "="*60)
print("SECTION 5: PROMPTING VS FINE-TUNING")
print("="*60)

comparison = {
    "Aspect": ["Time", "Data", "Cost", "Flexibility", "Knowledge", "When to use"],
    "Prompting": ["Instant", "None", "Free", "Flexible", "Learned patterns", "Quick direction"],
    "Fine-Tuning": ["Days", "Hundreds", "High", "Fixed", "Custom data", "New behavior"]
}

print("\n")
for i in range(len(comparison["Aspect"])):
    print(f"{comparison['Aspect'][i]:15} | {comparison['Prompting'][i]:20} | {comparison['Fine-Tuning'][i]:20}")

with open("prompting_vs_finetuning.json", "w", encoding='utf-8') as f:
    json.dump(comparison, f, indent=2)

# ============================================
# SECTION 6: CONTEXT-WINDOW
# ============================================

print("\n" + "="*60)
print("SECTION 6: CONTEXT-WINDOW NOTE")
print("="*60)

context_note = """CONTEXT WINDOW: Maximum tokens a model can see at once

Examples:
  BERT: 512 tokens (approximately 400 words)
  GPT-4: 128,000 tokens (approximately 100,000 words)
  Claude 3: 200,000 tokens (approximately 150,000 words)

WHY IT MATTERS:
  1. Cannot process text longer than limit
  2. Forgets early parts of long conversations
  3. More tokens means slower inference speed
  
REAL EXAMPLE:
  - Tell model: Always be professional
  - 400 tokens later: Tell me something funny
  - If window is 512, the be professional rule might be forgotten
"""

print(context_note)
with open("context_window_note.txt", "w", encoding='utf-8') as f:
    f.write(context_note)

# ============================================
# SECTION 7: FAILURE MODES
# ============================================

print("\n" + "="*60)
print("SECTION 7: MODEL FAILURE MODES")
print("="*60)

failure_modes = [
    {
        "name": "Ambiguous Reference",
        "example": "The trophy doesn't fit because it is too large. What is too large?",
        "why": "Pronouns can refer to multiple nouns - hard to disambiguate"
    },
    {
        "name": "Out of Knowledge",
        "example": "What happened in the 2035 Olympics?",
        "why": "Training data has a cutoff - model will hallucinate"
    },
    {
        "name": "Logic vs Reality",
        "example": "If all birds fly and penguins are birds, can they fly?",
        "why": "Real-world knowledge contradicts logical premise"
    },
    {
        "name": "Math Errors",
        "example": "What is 999,999 times 999,999?",
        "why": "Transformers are pattern-matchers, not calculators"
    }
]

print(f"\n{len(failure_modes)} Common Failure Modes:\n")
for i, failure in enumerate(failure_modes):
    print(f"{i+1}. {failure['name']}: {failure['why']}")

with open("failure_modes.json", "w", encoding='utf-8') as f:
    json.dump(failure_modes, f, indent=2)

# ============================================
# SECTION 8: RISKS
# ============================================

print("\n" + "="*60)
print("SECTION 8: RISKS AND LIMITATIONS")
print("="*60)

risks = """KEY LIMITATIONS OF TRANSFORMERS:

1. HALLUCINATIONS: Model generates confident but false information
2. CONTEXT LIMIT: Cannot process documents longer than context window
3. BIAS: Learns biases from training data
4. NO REASONING: Pattern matching, not logical deduction
5. SECURITY: Vulnerable to prompt injection attacks
6. COST: Large models are computationally expensive

RESPONSIBLE USE:
  - Verify facts independently
  - Use for drafts, not ground truth
  - Monitor outputs for bias
  - Implement human review for important decisions
  - Protect user data and privacy
  - Be transparent about limitations
"""

print(risks)
with open("risks_and_limitations.txt", "w", encoding='utf-8') as f:
    f.write(risks)

# ============================================
# SUMMARY
# ============================================

print("\n" + "="*60)
print("DEMONSTRATION COMPLETE")
print("="*60)

print("\nGenerated files:")
print("  - tokenization_results.json")
print("  - embedding_similarities.json")
print("  - prompting_vs_finetuning.json")
print("  - failure_modes.json")
print("  - context_window_note.txt")
print("  - risks_and_limitations.txt")
print("  - embedding_visualization.png")
print("  - transformer_architecture_diagram.png")

