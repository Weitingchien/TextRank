import numpy as np
import networkx as nx
import matplotlib.pyplot as plt

# 定義共現矩陣

def main():
    co_occurrence_matrix = np.array([[0, 3, 1, 1, 1],
                                [3, 0, 1, 1, 1],
                                [1, 1, 0, 0, 0],
                                [1, 1, 0, 0, 0],
                                [1, 1, 0, 0, 0]])

    # 創建空的無向圖
    G = nx.Graph()

    # 添加節點
    words = ['I', 'like', 'apple', 'orange', 'strawberry']
    G.add_nodes_from(words)

    # 添加邊
    for i in range(len(co_occurrence_matrix)):
        for j in range(i + 1, len(co_occurrence_matrix)):
            weight = co_occurrence_matrix[i][j]
            if weight > 0:
                G.add_edge(words[i], words[j], weight=weight)

    # 繪製無向圖
    pos = nx.spring_layout(G)  # 設置節點位置
    nx.draw_networkx_nodes(G, pos, node_color='lightblue', node_size=500)
    nx.draw_networkx_edges(G, pos, edge_color='gray')
    nx.draw_networkx_labels(G, pos, font_size=10, font_family='Arial')

    # 調整邊的粗細
    edge_weights = [G[u][v]['weight'] for u, v in G.edges()]
    nx.draw_networkx_edge_labels(G, pos, edge_labels=nx.get_edge_attributes(G, 'weight'),
                             font_size=8, font_color='red')

    plt.axis('off')
    plt.title('Co-occurrence Graph')
    plt.show()

    # 計算每個單詞的out degree
    out_degree_sum = co_occurrence_matrix.sum(axis=1)
    print(f'out_degree_sum: {out_degree_sum}')

    d = 0.85  # 阻尼係數
    max_iter = 100 # 最大迭代次數
    threshold = 1e-6  # 設定收斂閾值

    num_words = len(co_occurrence_matrix)
    print(f'num_words: {num_words}')
    word_scores = np.ones(num_words)
    print(f'word_scores: {word_scores}')

    # TextRank
    for k in range(max_iter):
        prev_scores = np.copy(word_scores)

        for i in range(num_words):
            print(f'co_occurrence({i}): {co_occurrence_matrix[:, i]}')
            incoming_scores = co_occurrence_matrix[:, i] / out_degree_sum
            print(f'incoming_scores({i}): {incoming_scores}')
            word_scores[i] = (1 - d) + d * np.sum(incoming_scores * prev_scores)
            print(word_scores[i])

        # 檢查是否收斂
        if np.sum(np.abs(word_scores - prev_scores)) < threshold:
            print(f'Converge at iteration: {k}')
            break
            
        # 輸出詞彙分數
        words = ['I', 'like', 'apple', 'orange', 'strawberry']
        for word, score in zip(words, word_scores):
            pass
            #print(f"{word}: {score}")

        # 計算詞彙分數的總和
        total_score = np.sum(word_scores)

        # 正規化詞彙分數
        normalized_scores = word_scores / total_score

    # 輸出正規化後的詞彙分數
    for word, score in zip(words, normalized_scores):
        print(f"{word}: {score}")




if __name__ == '__main__':
    main()  


