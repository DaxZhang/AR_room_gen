import json, copy, math, os
import numpy as np
import seaborn as sns
from sklearn.cluster import DBSCAN
from scipy.spatial.distance import pdist, squareform
from scipy.sparse import csr_matrix
from scipy.sparse.csgraph import connected_components
from sklearn.metrics import silhouette_score
import matplotlib.pyplot as plt
import matplotlib.patches as patches
from sklearn.cluster import DBSCAN
import os
from pathlib import Path

class BboxFINCH:
    def __init__(self, distance_metric='min_distance'):
        """
        Initialize FINCH clustering for 3D bounding boxes
        
        Parameters:
        - distance_metric: Either a string ('min_distance', 'iou') or a custom function
                          that computes distance between two bboxes
        """
        self.distance_metric = distance_metric
        self.partitions_ = []
        
    def _compute_distance(self, bbox1, bbox2):
        """Compute distance between two bboxes based on the specified metric"""
        if callable(self.distance_metric):
            return self.distance_metric(bbox1, bbox2)
        
        if self.distance_metric == 'min_distance':
            return self._min_distance(bbox1, bbox2)
        elif self.distance_metric == 'iou':
            return 1 - self._iou(bbox1, bbox2)  # Convert similarity to distance
        else:
            raise ValueError(f"Unknown distance metric: {self.distance_metric}")
    
    @staticmethod
    def _min_distance(bbox1, bbox2):
        """Calculate minimum Euclidean distance between two 3D boxes"""
        min1, max1 = np.array(bbox1['min']), np.array(bbox1['max'])
        min2, max2 = np.array(bbox2['min']), np.array(bbox2['max'])

        # Calculate separation in each dimension
        dist = np.maximum(0, np.maximum(min1 - max2, min2 - max1))
        return np.linalg.norm(dist)
    
    @staticmethod
    def _iou(bbox1, bbox2):
        """Calculate Intersection over Union (IoU) for 3D boxes"""
        min1, max1 = np.array(bbox1['min']), np.array(bbox1['max'])
        min2, max2 = np.array(bbox2['min']), np.array(bbox2['max'])
        
        # Calculate intersection volume
        intersection_min = np.maximum(min1, min2)
        intersection_max = np.minimum(max1, max2)
        intersection_dims = np.maximum(0, intersection_max - intersection_min)
        intersection_volume = np.prod(intersection_dims)
        
        # Calculate union volume
        vol1 = np.prod(max1 - min1)
        vol2 = np.prod(max2 - min2)
        union_volume = vol1 + vol2 - intersection_volume
        
        return intersection_volume / union_volume if union_volume > 0 else 0
    
    def _compute_adjacency_matrix(self, neighbor_indices_list):
        """
        改进的邻接矩阵计算，处理多个first neighbors的情况
        neighbor_indices_list: 每个元素是该物体的所有first neighbors的数组
        """
        n_samples = len(neighbor_indices_list)
        rows = []
        cols = []
        
        for i in range(n_samples):
            # 条件1: 连接该物体到它的所有first neighbors
            valid_nbrs = [j for j in neighbor_indices_list[i] if 0 <= j < n_samples]
            for j in valid_nbrs:
                rows.extend([i, j])  # 双向连接
                cols.extend([j, i])
            
            # 条件3: 连接共享任意first neighbor的物体
            if len(neighbor_indices_list[i]) > 0:
                # 获取该物体的所有first neighbors
                my_first_nbrs = set(neighbor_indices_list[i])
                
                # 查找其他以这些neighbors作为first neighbor的物体
                for other in range(n_samples):
                    if other == i:
                        continue
                    
                    other_first_nbrs = set(neighbor_indices_list[other])
                    if my_first_nbrs & other_first_nbrs:  # 有共享的first neighbor
                        rows.extend([i, other])
                        cols.extend([other, i])
        
        # 创建稀疏邻接矩阵
        # if max(rows + cols) >= n_samples:
        #     raise ValueError("Invalid index in adjacency matrix")
        data = np.ones(len(rows))
        adjacency = csr_matrix((data, (rows, cols)), shape=(n_samples, n_samples))
        
        return adjacency    
    
    def _get_first_neighbors(self, distances):
        """
        获取每个物体的first neighbors，处理距离为0的平局情况
        返回一个二维数组，每行包含该物体的所有first neighbors的索引
        """
        n_samples = distances.shape[0]
        neighbor_indices = []
        
        for i in range(n_samples):
            # 找到所有距离为最小值的邻居（可能不止一个）
            other_dists = np.delete(distances[i], i)
            if len(other_dists) == 0:
                # 如果只有一个物体（没有其他物体）
                neighbor_indices.append(np.array([], dtype=int))
                continue

            min_dist = np.min(other_dists)
            first_nbrs = np.where(distances[i] == min_dist)[0]
            first_nbrs = first_nbrs[first_nbrs != i]  # 排除自身
            
            if len(first_nbrs) == 0:
                # 如果没有其他物体（只有自身），保持空列表
                first_nbrs = np.array([], dtype=int)
            
            neighbor_indices.append(first_nbrs)
        
        return neighbor_indices

    def fit(self, bboxes):
        """
        Fit FINCH clustering to bounding boxes
        
        Parameters:
        - bboxes: List of bounding boxes in format:
                  [{"min": [x1,y1,z1], "max": [x2,y2,z2]}, ...]
        
        Returns:
        - self: Returns an instance of self
        """
        self.bboxes_ = bboxes
        n_samples = len(bboxes)
        if n_samples == 0:
            return self
        current_data = bboxes.copy()
        current_labels = np.arange(n_samples, dtype=np.int32)  # Initial labels
        original_labels = np.arange(n_samples, dtype=np.int32)
        
        while True:
            # Step 1: Compute pairwise distances and find first neighbors
            distances = np.zeros((len(current_data), len(current_data)))
            for i in range(len(current_data)):
                for j in range(i+1, len(current_data)):
                    dist = self._compute_distance(current_data[i], current_data[j])
                    distances[i, j] = dist
                    distances[j, i] = dist
            
            # Find first neighbor for each point (excluding self)
            neighbor_indices_list = self._get_first_neighbors(distances)
            
            # Step 2: Compute adjacency matrix
            adjacency = self._compute_adjacency_matrix(neighbor_indices_list)
            print(adjacency.todense())
            # Step 3: Find connected components (clusters)
            n_components, labels = connected_components(adjacency, directed=False)
            print(n_components,'  ', labels)
            # Map labels to original bboxes
            new_labels = np.empty(n_samples, dtype=np.int32)
            for new_label in range(n_components):
                indices = (labels == new_label).nonzero()
                new_labels[np.isin(original_labels, indices)] = new_label
            
            original_labels = new_labels

            # Store partition
            self.partitions_.append(original_labels.copy())
            
            #Finch算法是可进行递归进行层次聚类的，针对bounding box仍有bug，此处强制终止递归
            # Check stopping condition (only one cluster left)
            # if True:
            #     break
            if n_components == 1:
                break
                
            # Prepare for next iteration: Compute cluster representatives
            new_data = []
            for cluster_id in range(n_components):
                cluster_mask = (original_labels == cluster_id)
                cluster_bboxes = [bboxes[i] for i in np.where(cluster_mask)[0]]
                
                # Compute mean bbox as cluster representative
                min_coords = np.mean([b['min'] for b in cluster_bboxes], axis=0)
                max_coords = np.mean([b['max'] for b in cluster_bboxes], axis=0)
                new_data.append({"min": min_coords.tolist(), "max": max_coords.tolist()})
            
            current_data = new_data
            current_labels = np.arange(n_components)
            
        return self
    
    def get_partitions(self):
        """Get all hierarchical partitions"""
        return self.partitions_
    
    def get_optimal_partition(self, n_clusters=None):
        #没进行层次聚类这个函数不用管
        """
        Get optimal partition either automatically or by specifying number of clusters
        
        Parameters:
        - n_clusters: Desired number of clusters (if None, selects partition with highest silhouette score)
        
        Returns:
        - Array of cluster labels
        """
        if n_clusters is not None:
            # Find the partition with closest number of clusters to requested
            closest_part = None
            min_diff = float('inf')
            
            for part in self.partitions_:
                n_part_clusters = len(np.unique(part))
                diff = abs(n_part_clusters - n_clusters)
                if diff < min_diff:
                    min_diff = diff
                    closest_part = part
            
            return closest_part
        else:
            # Automatically select partition using silhouette score
            
            # Need to compute distance matrix for all bboxes for silhouette score
            distance_matrix = np.zeros((len(self.bboxes_), len(self.bboxes_)))
            for i in range(len(self.bboxes_)):
                for j in range(i+1, len(self.bboxes_)):
                    dist = self._compute_distance(self.bboxes_[i], self.bboxes_[j])
                    distance_matrix[i, j] = dist
                    distance_matrix[j, i] = dist
            
            best_score = -1
            best_part = None
            
            for part in self.partitions_:
                n_clusters = len(np.unique(part))
                if n_clusters == 1 or n_clusters == len(self.bboxes_):
                    continue  # Skip trivial cases
                
                score = silhouette_score(distance_matrix, part, metric='precomputed')
                if score > best_score:
                    best_score = score
                    best_part = part
            
            if best_part is not None: return best_part
            if len(self.partitions_) > 0 : return self.partitions_[0] # trivial case: only one cluster
            return None
        
def get_file_num(folder, prefix):
    os.makedirs(folder, exist_ok=True)  # 确保文件夹存在
    files = os.listdir(folder)
    count = sum(1 for f in files if f.startswith(prefix) )
    return count

def draw_room_clusters(room_dict, labels):
    if len(labels) <5:
        return None#数量少的，直接不画
    obj_list = room_dict.get('objList', [])
    assert len(obj_list) == len(labels), "标签数量应与物体数量相同"

    # cmap = plt.colormaps['tab20'].resampled(np.max(labels) + 2)
    cmap = plt.cm.get_cmap('tab20', np.max(labels) + 2)
    fig, ax = plt.subplots(figsize=(10, 8))
    ax.set_title("Room Object Clusters (Top-Down XY View)")

    all_x, all_z = [], []

    for obj, label in zip(obj_list, labels):
        bbox = obj['bbox']
        min_pt, max_pt = bbox['min'], bbox['max']
        x, z = min_pt[0], min_pt[2]
        w, h = max_pt[0] - min_pt[0], max_pt[2] - min_pt[2]
        if w < 0: x, w = x + w, -w
        if h < 0: z, h = z + h, -h

        all_x.extend([x, x + w])
        all_z.extend([z, z + h])

        color = 'gray' if label == -1 else cmap(label)
        rect = patches.Rectangle((x, z), w, h, linewidth=1.2,
                                 edgecolor=color, facecolor=color, alpha=0.5)
        ax.add_patch(rect)

        if w > 0.1 and h > 0.1:
            ax.text(x + w/2, z + h/2,
                    obj.get('coarseSemantic', ''),
                    fontsize=6, color='black',
                    ha='center', va='center',
                    clip_on=True)

    shape = room_dict.get("roomShape", [])
    if shape:
        room_poly = patches.Polygon(shape, closed=True, fill=False,
                                    edgecolor='black', linewidth=2, linestyle='--')
        ax.add_patch(room_poly)
        xs, zs = zip(*shape)
        all_x.extend(xs)
        all_z.extend(zs)

    # 设置可视范围 + 留边距
    if all_x and all_z:
        x_min, x_max = min(all_x), max(all_x)
        z_min, z_max = min(all_z), max(all_z)
        pad_x = (x_max - x_min) * 0.1
        pad_z = (z_max - z_min) * 0.1
        ax.set_xlim(x_min - pad_x, x_max + pad_x)
        ax.set_ylim(z_min - pad_z, z_max + pad_z)

    ax.set_aspect('equal')
    ax.invert_yaxis()  # 上为正方向
    plt.xlabel("X")
    plt.ylabel("Z")
    plt.grid(True)
    plt.tight_layout()
    if Debug:
        plt.show()
    return fig

def save_figure(fig, path='./runs/layout_cluster_vis/', prefix='',end_with=''):
    num = get_file_num(path, prefix)
    path = os.path.join(path, f"{prefix}{num:04d}{end_with}.png")
    fig.savefig(path, dpi=300)
    print(f"Figure saved to {path}")

def expand_bbox(min_pt, max_pt, scale=1.2):
    min_pt = np.array(min_pt)
    max_pt = np.array(max_pt)
    center = (min_pt + max_pt) / 2
    half_size = (max_pt - min_pt) / 2 * scale
    new_min = center - half_size
    new_max = center + half_size
    return new_min, new_max

def bbox_iou(min1, max1, min2, max2):
    min1 = np.array(min1)
    max1 = np.array(max1)
    min2 = np.array(min2)
    max2 = np.array(max2)
    # print("min1, max1, min2, max2",min1, max1, min2, max2)
    inter_min = np.maximum(min1, min2)
    inter_max = np.minimum(max1, max2)
    inter_dims = np.maximum(0, inter_max - inter_min)
    inter_vol = np.prod(inter_dims)
    print("inter_min, inter_max, inter_dims, inter_vol",inter_min, inter_max, inter_dims, inter_vol)
    vol1 = np.prod(max1 - min1)
    vol2 = np.prod(max2 - min2)
    union = vol1 + vol2 - inter_vol
    # print("vol1, vol2, union",vol1, vol2, union)

    return inter_vol / union if union > 0 else 0

def expand_bbox_iou(bbox1, bbox2, scale=1.2):
    min1, max1 = np.array(bbox1['min']), np.array(bbox1['max'])
    min2, max2 = np.array(bbox2['min']), np.array(bbox2['max'])
    min_expand1, max_expand1 = expand_bbox(min1, max1, scale)
    min_expand2, max_expand2 = expand_bbox(min2, max2, scale)
    iou = bbox_iou(min_expand1, max_expand1, min_expand2, max_expand2)
    return iou

def expand_bbox_min_distance(bbox1, bbox2, scale=1.2):
    min1, max1 = np.array(bbox1['min']), np.array(bbox1['max'])
    min2, max2 = np.array(bbox2['min']), np.array(bbox2['max'])
    min_expand1, max_expand1 = expand_bbox(min1, max1, scale)
    min_expand2, max_expand2 = expand_bbox(min2, max2, scale)
    dist = np.maximum(0, np.maximum(min_expand1 - max_expand2, min_expand2 - max_expand1))
    return np.linalg.norm(dist)

def bbox_min_distance(bbox1, bbox2):
    """计算两个 bbox（3D box）之间的最小欧氏距离"""
    min1, max1 = np.array(bbox1['min']), np.array(bbox1['max'])
    min2, max2 = np.array(bbox2['min']), np.array(bbox2['max'])

    # 计算在每一维的间距
    dist = np.maximum(0, np.maximum(min1 - max2, min2 - max1))
    return np.linalg.norm(dist)

def cluster_objects_by_bbox(room_dict, eps=0.1, min_samples=2,method='bbox_min_distance',expand_ratio=0.3):
    """
    使用基于最小 bbox 距离的 DBSCAN 对房间中的对象聚类
    method: 'bbox_min_distance', 'bbox_expand_iou','expand_bbox_min_distance'
    """
    def distance(bbox1, bbox2):
        if method == 'bbox_min_distance':
            return bbox_min_distance(bbox1, bbox2)
        elif method == 'bbox_expand_iou':#效果差
            return expand_bbox_iou(bbox1, bbox2,scale=expand_ratio)
        elif method == 'expand_bbox_min_distance':
            return expand_bbox_min_distance(bbox1, bbox2,scale=expand_ratio)
    obj_list = room_dict.get('objList', [])
    n = len(obj_list)
    if n == 0:
        return [], np.array([[]])

    # 构造距离矩阵
    dist_matrix = np.zeros((n, n))
    for i in range(n):
        for j in range(i + 1, n):
            d = distance(obj_list[i]['bbox'], obj_list[j]['bbox'])
            dist_matrix[i, j] = dist_matrix[j, i] = d
    print("mean_dist",np.mean(dist_matrix),"\n","max_dist",np.max(dist_matrix))
    # if Debug:
    #     print("dist_matrix",dist_matrix)
    #     plt.imshow(dist_matrix, cmap='viridis')
    #     plt.colorbar()
    #     plt.show()
    # 用 sklearn 的 DBSCAN，并传入预计算距离矩阵
    clustering = DBSCAN(eps=eps, min_samples=min_samples, metric='precomputed')
    labels = clustering.fit_predict(dist_matrix)
    print(f"Clustering result: {set(labels)}")
    # 返回每个 object 的聚类标签（与 objList 同序）
    return labels, dist_matrix

# silhouette_score_original = silhouette_score
# def silhouette_score(*args, **kwargs):
#     try:
#         return silhouette_score_original(*args, **kwargs)
#     except:
#         return 1

global Debug
if __name__ == '__main__':
    Debug = False
    folder = Path('./data/scene_layout_json/')
    json_files = sorted(folder.glob("*.json")) 
    if Debug:
        json_files = ['data/scene_layout_json/01e1d6b2-e3b3-4eb4-9969-b23088fab6a0.json']

    cluster_eval = {"DBSCAN":[],"expand_DBSCAN":[],"BboxFINCH":[], "BboxFINCHOptimal": []}

    for json_file in json_files:
        
        with open(json_file, 'r', encoding='utf-8') as f:
            try:
                data = json.load(f)
            except json.JSONDecodeError:
                print(f"Error reading {json_file.name}, skipping.")
        rooms=data['rooms']


        for room in rooms:
            try:
                labels0, original_dist_matrix = cluster_objects_by_bbox(room,eps=0.2,min_samples=2,method='bbox_min_distance')
                labels1, expand_dist_matrix = cluster_objects_by_bbox(room,eps=0.2,min_samples=2,method='expand_bbox_min_distance',expand_ratio=1.4)
                obj_bboxes = [obj['bbox'] for obj in room['objList']]
                finch = BboxFINCH(distance_metric='min_distance')
                finch.fit(obj_bboxes)
                partitions = finch.get_partitions()
                labels2 = partitions[0]
                print("Partitions:", partitions)
                labels3 = finch.get_optimal_partition() #暂时有错

                #定量指标 轮廓系数
                DBSCAN_score = silhouette_score(original_dist_matrix, labels0, metric='precomputed')
                expand_DBSCAN_score = silhouette_score(original_dist_matrix, labels1, metric='precomputed')
                Finch_score = silhouette_score(original_dist_matrix, labels2, metric='precomputed')
                Finch_optimal_score = silhouette_score(original_dist_matrix, labels3, metric='precomputed')
                cluster_eval['DBSCAN'].append(DBSCAN_score)
                cluster_eval['expand_DBSCAN'].append(expand_DBSCAN_score)
                cluster_eval['BboxFINCH'].append(Finch_score)
                cluster_eval['BboxFINCHOptimal'].append(Finch_optimal_score)
                drawing = True #控制是否绘制图像，保存在./runs/layout_cluster_vis/
                if drawing:
                    fig1 = draw_room_clusters(room, labels1)
                    fig2 = draw_room_clusters(room, labels2)
                    fig3 = draw_room_clusters(room, labels3)
                    # if labels3 is not None:
                    #     fig3 = draw_room_clusters(room, labels3)
                    if fig1 is not None and fig2 is not None:
                        if Debug:
                            pass
                        else:
                            save_figure(fig1, prefix=f"clusterVis_{json_file.stem}",end_with="_expand_DBSCAN")
                            save_figure(fig2, prefix=f"clusterVis_{json_file.stem}",end_with="_FINCH")
                            if labels3 is not None:
                                save_figure(fig3, prefix=f"clusterVis_{json_file.stem}",end_with="_FINCH_Optimal")
                        # plt.show()
            except Exception as e:
                print(f"Error processing , skipping. {e}")
                # raise e
        
        print(f"DBSCAN_score_mean:{np.mean(cluster_eval['DBSCAN'])},expand_DBSCAN_score_mean:{np.mean(cluster_eval['expand_DBSCAN'])},BboxFINCH_score_mean:{np.mean(cluster_eval['BboxFINCH'])} ")


    #打印/输出三种方法的轮廓系数的统计量
    print(f"DBSCAN_score_std:{np.std(cluster_eval['DBSCAN'])},expand_DBSCAN_score_std:{np.std(cluster_eval['expand_DBSCAN'])},BboxFINCH_score_std:{np.std(cluster_eval['BboxFINCH'])} ")
    plt.close('all')
    sns.kdeplot(cluster_eval['DBSCAN'], fill=True, label='DBSCAN')
    sns.kdeplot(cluster_eval['expand_DBSCAN'], fill=True, label='expand_DBSCAN')
    sns.kdeplot(cluster_eval['BboxFINCH'], fill=True, label='FINCH')
    sns.kdeplot(cluster_eval['BboxFINCHOptimal'], fill=True, label='FINCHOptimal')
    plt.xlabel('Value')
    plt.ylabel('Density')
    plt.title('silhouette score Distribution Comparison')
    plt.legend()
    plt.show()
    plt.savefig('kde.png')

