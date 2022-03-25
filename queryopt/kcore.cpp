#include <algorithm>
#include "kcore.h"
void crossUpdate_for_kcore(BiGraph& g, int alpha, int k_x, vid_t v) {
	for (int beta = k_x; beta > 0; beta--) {
		if (g.right_index[v][beta] < alpha) {
			g.right_index[v][beta] = alpha;
		}
		else {
			break;
		}
	}
}
void print_index_detail(BiGraph& g) {
	for (vid_t u = 0; u < g.num_v1; u++) {
		cout << "u" << u + 1 << " : ";
		for (int i = 1; i < g.left_index[u].size(); i++) {
			cout << g.left_index[u][i] << " ";
		}
		cout << endl;
	}
	for (vid_t v = 0; v < g.num_v2; v++) {
		cout << "v" << v + 1 << " : ";
		for (int i = 1; i < g.right_index[v].size(); i++) {
			cout << g.right_index[v][i] << " ";
		}
		cout << endl;
	}
}
int coreIndexKCore(BiGraph& g) {
	int left_degree_max = 0;
	for (int i = 0; i < g.getV1Num(); i++) {
		if (left_degree_max < g.getV1Degree(i)) left_degree_max = g.getV1Degree(i);
	}
	int right_degree_max = 0;
	for (int i = 0; i < g.getV2Num(); i++) {
		if (right_degree_max < g.getV2Degree(i)) right_degree_max = g.getV2Degree(i);
	}
	// init g's max degree and index
	g.v1_max_degree = left_degree_max;
	g.v2_max_degree = right_degree_max;
	g.left_index.resize(g.getV1Num());
	g.right_index.resize(g.getV2Num());
	g.left_delete.resize(g.getV1Num());
	g.right_delete.resize(g.getV2Num());
	fill_n(g.left_delete.begin(), g.left_delete.size(), false);
	fill_n(g.right_delete.begin(), g.right_delete.size(), false);
	for (int i = 0; i < g.getV1Num(); i++) {
		g.left_index[i].resize(g.getV1Degree(i) + 1);
		fill_n(g.left_index[i].begin(), g.left_index[i].size(), 0);
	}
	for (int i = 0; i < g.getV2Num(); i++) {
		g.right_index[i].resize(g.getV2Degree(i) + 1);
		fill_n(g.right_index[i].begin(), g.right_index[i].size(), 0);
	}
	//int km = maxkcorenumber_test_optimal(g);
	//cout << "max kcore number: " << km << endl;
	int beta_s = 0;
	for (int left_k = 1; left_k <= g.v1_max_degree; left_k++) {
		alphaCopyPeel_for_kcore(left_k, g);
		if (PRINT_INDEX_DETAIL) {
			print_index_detail(g);
			cout << endl;
		}
		beta_s = 0;
		for (vid_t u = 0; u < g.num_v1; u++) {
			if (g.degree_v1[u] <= left_k) continue;
			int right_k = g.left_index[u][left_k];
			if (beta_s < right_k) beta_s = right_k;
		}
		if (beta_s <= left_k) break;
	}
	// restore g
	fill_n(g.left_delete.begin(), g.left_delete.size(), false);
	g.v1_max_degree = left_degree_max;
	for (vid_t u = 0; u < g.num_v1; u++) {
		g.degree_v1[u] = g.neighbor_v1[u].size();
	}
	fill_n(g.right_delete.begin(), g.right_delete.size(), false);
	g.v2_max_degree = right_degree_max;
	for (vid_t v = 0; v < g.num_v2; v++) {
		g.degree_v2[v] = g.neighbor_v2[v].size();
	}

	if (PRINT_INDEX_DETAIL) {
		cout << "inverse" << endl;
		cout << endl;
	}
	inv(g);
	for (int left_k = 1; left_k <= beta_s; left_k++) {
		alphaCopyPeel_for_kcore(left_k, g);
		if (PRINT_INDEX_DETAIL) {
			inv(g);
			print_index_detail(g);
			cout << endl;
			inv(g);
		}
	}
	inv(g);
	// restore g
	fill_n(g.left_delete.begin(), g.left_delete.size(), false);
	g.v1_max_degree = left_degree_max;
	for (vid_t u = 0; u < g.num_v1; u++) {
		g.degree_v1[u] = g.neighbor_v1[u].size();
	}
	fill_n(g.right_delete.begin(), g.right_delete.size(), false);
	g.v2_max_degree = right_degree_max;
	for (vid_t v = 0; v < g.num_v2; v++) {
		g.degree_v2[v] = g.neighbor_v2[v].size();
	}
	return beta_s;
}

void alphaCopyPeel_for_kcore(int left_k, BiGraph& g) {
	int dd_;
	int pre_left_k_ = left_k - 1;
	vector<bool> left_deletion_next_round;
	vector<bool> right_deletion_next_round;
	vector<int> left_degree_next_round;
	vector<int> right_degree_next_round;
	vector<vid_t> left_vertices_to_be_peeled;
	vector<vid_t> right_vertices_to_be_peeled;
	for (vid_t u = 0; u < g.getV1Num(); u++) {
		if (g.degree_v1[u] < left_k && !g.left_delete[u]) {
			left_vertices_to_be_peeled.push_back(u);
		}
	}
	int right_remain_nodes_num = g.num_v2;
	vector<vid_t> right_remain_nodes; right_remain_nodes.resize(g.num_v2);
	for (int i = 0; i < right_remain_nodes.size(); i++) {
		right_remain_nodes[i] = i;
	}
	int right_remain_nodes_tmp_num = 0;
	vector<vid_t> right_remain_nodes_tmp; right_remain_nodes_tmp.resize(g.num_v2);
	bool update_flag = false;
	for (int right_k = 1; right_k <= g.v2_max_degree + 1; right_k++) {
		if (right_k - 1 > 0) {
			update_flag = true;
		}
		int pre_ = right_k - 1;
		bool stop = true;
		right_remain_nodes_tmp_num = 0;
		for (int i = 0; i < right_remain_nodes_num; i++) {
			vid_t v = right_remain_nodes[i];
			if (!g.right_delete[v]) {
				stop = false;
				right_remain_nodes_tmp[right_remain_nodes_tmp_num] = v;
				right_remain_nodes_tmp_num++;
				if (g.degree_v2[v] < right_k) {
					right_vertices_to_be_peeled.push_back(v);
				}
			}
		}
		swap(right_remain_nodes, right_remain_nodes_tmp);
		right_remain_nodes_num = right_remain_nodes_tmp_num;
		if (stop) break;
		while (!left_vertices_to_be_peeled.empty() || !right_vertices_to_be_peeled.empty()) {
			// peel left
			int oo_ = left_vertices_to_be_peeled.size();
			for (int j = 0; j < oo_; j++) {
				vid_t u = left_vertices_to_be_peeled[j];
				if (g.left_delete[u]) continue;
				vector<vid_t>& tmp_neigh_ = g.neighbor_v1[u];
				int ss = tmp_neigh_.size();
				for (int k = 0; k < ss; k++) {
					vid_t v = tmp_neigh_[k];
					if (g.right_delete[v]) continue;
					dd_ = --g.degree_v2[v];
					if (update_flag && dd_ == 0) {
						crossUpdate_for_kcore(g, left_k, pre_, v);
						g.right_delete[v] = true;
					}
					if (dd_ == pre_) {
						right_vertices_to_be_peeled.push_back(v);
					}
				}
				g.degree_v1[u] = 0;
				g.left_delete[u] = true;
				if (update_flag) {
					g.left_index[u][left_k] = pre_;
				}
			}
			left_vertices_to_be_peeled.clear();
			// peel right
			oo_ = right_vertices_to_be_peeled.size();
			for (int j = 0; j < oo_; j++) {
				vid_t v = right_vertices_to_be_peeled[j];
				if (g.right_delete[v]) continue;
				vector<vid_t>& tmp_neigh_ = g.neighbor_v2[v];
				int ss = tmp_neigh_.size();
				for (int k = 0; k < ss; k++) {
					vid_t u = tmp_neigh_[k];
					if (g.left_delete[u]) continue;
					dd_ = --g.degree_v1[u];
					if (update_flag && dd_ == 0) {
						g.left_index[u][left_k] = pre_;
						g.left_delete[u] = true;
					}
					if (dd_ == pre_left_k_) {
						left_vertices_to_be_peeled.push_back(u);
					}
				}
				g.degree_v2[v] = 0;
				g.right_delete[v] = true;
				if (update_flag) {
					crossUpdate_for_kcore(g, left_k, pre_, v);
				}
			}
			right_vertices_to_be_peeled.clear();
		}
		if (right_k == 1) {
			left_degree_next_round = g.degree_v1;
			right_degree_next_round = g.degree_v2;
			left_deletion_next_round = g.left_delete;
			right_deletion_next_round = g.right_delete;
		}
	}
	g.degree_v1 = left_degree_next_round;
	g.degree_v2 = right_degree_next_round;
	g.left_delete = left_deletion_next_round;
	g.right_delete = right_deletion_next_round;
	g.v1_max_degree = 0;
	g.v2_max_degree = 0;
	for (vid_t u = 0; u < g.degree_v1.size(); u++) {
		if (g.v1_max_degree < g.degree_v1[u]) g.v1_max_degree = g.degree_v1[u];
	}
	for (vid_t v = 0; v < g.degree_v2.size(); v++) {
		if (g.v2_max_degree < g.degree_v2[v]) g.v2_max_degree = g.degree_v2[v];
	}
}
