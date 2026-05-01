import numpy as np
import torch
import torch.nn.functional as F
import matplotlib.pyplot as plt
import umap


# =====================================================
# 1. NORMALIZE EMBEDDINGS
# =====================================================

def normalize_embeddings(embeddings):

    emb = torch.tensor(embeddings, dtype=torch.float32)
    emb = F.normalize(emb, dim=1)

    return emb


# =====================================================
# 2. COSINE SIMILARITY MATRIX
# =====================================================

def similarity_matrix(embeddings):

    emb = normalize_embeddings(embeddings)

    sim = emb @ emb.T

    return sim.numpy()


# =====================================================
# 3. SELF RETRIEVAL ACCURACY
# =====================================================

def self_retrieval_accuracy(embeddings):

    emb = normalize_embeddings(embeddings)

    sim = emb @ emb.T

    preds = sim.argmax(dim=1)

    labels = torch.arange(len(embeddings))

    acc = (preds == labels).float().mean().item()

    print("\n[Self Retrieval Accuracy]")
    print(f"{acc:.4f}")

    return acc


# =====================================================
# 4. SAME USER RETRIEVAL (OPTIONAL)
# =====================================================

def same_user_retrieval_accuracy(embeddings, user_ids):

    emb = normalize_embeddings(embeddings)

    sim = emb @ emb.T

    user_ids = np.array(user_ids)

    correct = 0

    for i in range(len(user_ids)):

        sim[i, i] = -1  # ignore self

        nn_idx = sim[i].argmax().item()

        if user_ids[i] == user_ids[nn_idx]:
            correct += 1

    acc = correct / len(user_ids)

    print("\n[Same User Retrieval Accuracy]")
    print(f"{acc:.4f}")

    return acc


# =====================================================
# 5. UMAP VISUALIZATION
# =====================================================

def plot_embeddings(embeddings, labels=None):

    reducer = umap.UMAP(
        n_components=2,
        random_state=42
    )

    reduced = reducer.fit_transform(embeddings)

    plt.figure(figsize=(10, 7))

    if labels is None:
        plt.scatter(reduced[:, 0], reduced[:, 1], s=5)
    else:
        plt.scatter(
            reduced[:, 0],
            reduced[:, 1],
            c=labels,
            cmap="tab20",
            s=5
        )

    plt.title("Keystroke Embedding Space (UMAP)")
    plt.xlabel("Dim 1")
    plt.ylabel("Dim 2")

    plt.show()


# =====================================================
# 6. FULL EVALUATION PIPELINE
# =====================================================

def evaluate_embeddings(embeddings, user_ids=None):

    print("\n==============================")
    print(" EMBEDDING EVALUATION")
    print("==============================\n")

    embeddings = np.array(embeddings)

    # 1. self retrieval
    self_acc = self_retrieval_accuracy(embeddings)

    # 2. same user retrieval (if available)
    user_acc = None
    if user_ids is not None:
        user_acc = same_user_retrieval_accuracy(embeddings, user_ids)

    # 3. similarity matrix
    sim = similarity_matrix(embeddings)
    print("\nSimilarity matrix shape:", sim.shape)

    # 4. visualization
    plot_embeddings(embeddings, user_ids)

    print("\n==============================")
    print(" DONE")
    print("==============================\n")

    return {
        "self_retrieval_accuracy": self_acc,
        "user_retrieval_accuracy": user_acc
    }


# =====================================================
# 7. EXAMPLE USAGE
# =====================================================

if __name__ == "__main__":

    # Example loading
    embeddings = np.load("test_embeddings.npy")

    # OPTIONAL: if you have user IDs aligned with embeddings
    # user_ids = np.load("test_user_ids.npy")
    user_ids = None

    metrics = evaluate_embeddings(embeddings, user_ids)

    print(metrics)