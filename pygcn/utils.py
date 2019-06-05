import numpy as np
import scipy.sparse as sp
import torch


def encode_onehot(labels):
    classes = set(labels)
    classes_dict = {c: np.identity(len(classes))[i, :] for i, c in
                    enumerate(classes)}
    labels_onehot = np.array(list(map(classes_dict.get, labels)),
                             dtype=np.int32)
    return labels_onehot


def load_data(path="../data/cora/", dataset="cora"):
    """Load citation network dataset (cora only for now)"""
    print('Loading {} dataset...'.format(dataset))

    idx_features_labels = np.genfromtxt("{}{}.content".format(path, dataset),
                                        dtype=np.dtype(str))
    features = sp.csr_matrix(idx_features_labels[:, 1:-1], dtype=np.float32)
    labels = encode_onehot(idx_features_labels[:, -1])
    
    # build graph
    idx = np.array(idx_features_labels[:, 0], dtype=np.int32)
    # 取第一列paper的index
    idx_map = {j: i for i, j in enumerate(idx)}
    # 将index映射为整型编码
    edges_unordered = np.genfromtxt("{}{}.cites".format(path, dataset),
                                    dtype=np.int32)
    # 读取引用关系
    edges = np.array(list(map(idx_map.get, edges_unordered.flatten())),
                     dtype=np.int32).reshape(edges_unordered.shape)
    # 将edge list展平后映射idx_map，再reshape为edge list
    adj = sp.coo_matrix((np.ones(edges.shape[0]), (edges[:, 0], edges[:, 1])),
                        shape=(labels.shape[0], labels.shape[0]),
                        dtype=np.float32)
    # 将edge list建立为coodinate稀疏矩阵

    # build symmetric adjacency matrix
    adj = adj + adj.T.multiply(adj.T > adj) - adj.multiply(adj.T > adj)
    # 建立对称矩阵，k神的奇淫巧技！
    
    features = normalize(features)
    # 按行进行标准化：元素/行和
    adj = normalize(adj + sp.eye(adj.shape[0]))
    #adj矩阵自环后标准化
    
    idx_train = range(140)
    idx_val = range(200, 500)
    idx_test = range(500, 1500)
    #这也太随意了吧...

    features = torch.FloatTensor(np.array(features.todense()))
    # 将稀疏矩阵转化为torch tensor
    
    labels = torch.LongTensor(np.where(labels)[1])
    # 将onehot后的标签，又转化为整型编码
    # 开始感到反感，感觉没必要搞得这么麻烦
    adj = sparse_mx_to_torch_sparse_tensor(adj)
    # 转scipy稀疏矩阵为torch tensor稀疏矩阵
    
    idx_train = torch.LongTensor(idx_train)
    idx_val = torch.LongTensor(idx_val)
    idx_test = torch.LongTensor(idx_test)

    return adj, features, labels, idx_train, idx_val, idx_test

# 按行进行标准化：元素/行和
def normalize(mx):
    """Row-normalize sparse matrix"""
    rowsum = np.array(mx.sum(1))
    # 按行求和
    r_inv = np.power(rowsum, -1).flatten()
    # 求倒数后展开成一维向量
    r_inv[np.isinf(r_inv)] = 0.
    # 将inf转为0.
    r_mat_inv = sp.diags(r_inv)
    # 转为对角矩阵
    mx = r_mat_inv.dot(mx)
    # 将mx每行都处以行和
    # 矢量运算，速度更快，nb
    return mx


def accuracy(output, labels):
    preds = output.max(1)[1].type_as(labels)
    correct = preds.eq(labels).double()
    correct = correct.sum()
    return correct / len(labels)

# 转scipy稀疏矩阵为torch tensor稀疏矩阵，是个狼人
def sparse_mx_to_torch_sparse_tensor(sparse_mx):
    """Convert a scipy sparse matrix to a torch sparse tensor."""
    sparse_mx = sparse_mx.tocoo().astype(np.float32)
    #转为coodinate稀疏矩阵
    indices = torch.from_numpy(
        np.vstack((sparse_mx.row, sparse_mx.col)).astype(np.int64))
    values = torch.from_numpy(sparse_mx.data)
    shape = torch.Size(sparse_mx.shape)
    return torch.sparse.FloatTensor(indices, values, shape)
