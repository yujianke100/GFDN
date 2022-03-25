#pragma once
#ifndef __BIGRAPH_H
#define __BIGRAPH_H

#include <stdio.h>
#include <stdlib.h>
#include <string>
#include <vector>
#include <unordered_set>
#include "utility.h"

struct bicore_index_block {
	std::vector<vid_t> nodeset;
	bicore_index_block* next = NULL;
};

//struct bicore_index_block {
//	std::unordered_set<vid_t> nodeset;
//	bicore_index_block* next = NULL;
//};

struct bicore_index_block_dual_pointer {
	std::unordered_set<vid_t> nodeset;
	bicore_index_block_dual_pointer* horizontal_pointer = NULL;
	bicore_index_block_dual_pointer* vertical_pointer = NULL;
};

class Edge
{
public:
	Edge(int u_, int v_) { u = u_; v = v_; }
	bool operator<(const Edge &other) const
	{
		if (u == other.u)
			return v < other.v;
		return u < other.u;
	}

	int u;
	int v;
};


class DegreeNode
{
public:
	int id;
	int degree;
};

class BiGraph
{

public:

	BiGraph(int* inputA2, int D1, int D2, int n1, int n2);
	BiGraph(std::string dir);
	BiGraph();
	~BiGraph() {}

	void addEdge(vid_t u, vid_t v);
	void deleteEdge(vid_t u, vid_t v);
	bool isEdge(vid_t u, vid_t v);
	num_t getV1Num() { return num_v1; }
	num_t getV2Num() { return num_v2; }
	num_t getV1Degree(vid_t u) { return degree_v1[u]; }
	num_t getV2Degree(vid_t u) { return degree_v2[u]; }
	std::vector<vid_t> & getV2Neighbors(vid_t u) { return neighbor_v2[u]; }
	std::vector<vid_t> & getV1Neighbors(vid_t u) { return neighbor_v1[u]; }
	void print();
	void print(bool hash);
	void printSum();
	void printCout();

public:

	void init(unsigned int num_v1, unsigned int num_v2);
	void loadGraph(int* inputA2, int D1, int D2, int n1, int n2);
	void loadGraph(std::string dir);
	void compressGraph();

	std::string dir;
	num_t num_v1;
	num_t num_v2;
	num_t num_edges;

	std::vector<std::vector<vid_t>> neighbor_v1;
	std::vector<std::vector<vid_t>> neighbor_v2;

	std::vector<std::unordered_set<vid_t>> neighborHash_v1;
	std::vector<std::unordered_set<vid_t>> neighborHash_v2;

	std::vector<int> degree_v1;
	std::vector<int> degree_v2;

	std::vector<num_t> core_v1;
	std::vector<num_t> core_v2;

public:

	//KKCore index left (x,*) right (*,x)
	std::vector<std::vector<int>> left_index;
	std::vector<std::vector<int>> right_index;
	int v1_max_degree;
	int v2_max_degree;
	std::vector<bool> left_delete;
	std::vector<bool> right_delete;
	// for dynamic update
	std::vector<std::vector<int>> left_index_old;
	std::vector<std::vector<int>> right_index_old;
	//BiGraph operator=(const BiGraph& g);
	int delta = -1;

public:
	int get_left_index_with_fixed_left_k(vid_t u, int left_k);
	//BiGraph& operator=(const BiGraph& g_);
};

extern void build_bicore_index(BiGraph&g, std::vector<std::vector<bicore_index_block*>>& bicore_index_u, std::vector<std::vector<bicore_index_block*>>& bicore_index_v);

extern void build_bicore_index_space_saver(BiGraph&g, std::vector<std::vector<bicore_index_block*>>& bicore_index_u, std::vector<std::vector<bicore_index_block*>>& bicore_index_v);

extern void build_bicore_index_space_saver_dual_pointer(BiGraph&g, std::vector<std::vector<bicore_index_block_dual_pointer*>>& bicore_index_u, std::vector<std::vector<bicore_index_block_dual_pointer*>>& bicore_index_v);

extern void retrieve_via_bicore_index(BiGraph& g, std::vector<std::vector<bicore_index_block*>>& bicore_index_u, std::vector<std::vector<bicore_index_block*>>& bicore_index_v,
	std::vector<bool>& left_node, std::vector<bool>& right_node, int alpha, int beta);

extern void retrieve_via_bicore_index_space_saver(BiGraph& g, std::vector<std::vector<bicore_index_block*>>& bicore_index_u, std::vector<std::vector<bicore_index_block*>>& bicore_index_v,
	std::vector<bool>& left_node, std::vector<bool>& right_node, int alpha, int beta);

extern void retrieve_via_bicore_index_space_saver_dual_pointer(BiGraph& g, std::vector<std::vector<bicore_index_block_dual_pointer*>>& bicore_index_u, std::vector<std::vector<bicore_index_block_dual_pointer*>>& bicore_index_v,
	std::vector<bool>& left_node, std::vector<bool>& right_node, int alpha, int beta);

extern void retrieve_via_bicore_index_inverse(BiGraph& g, std::vector<std::vector<bicore_index_block*>>& bicore_index_u, std::vector<std::vector<bicore_index_block*>>& bicore_index_v,
	std::vector<bool>& left_node, std::vector<bool>& right_node, int alpha, int beta);

extern void inv(BiGraph& g);

#endif  /* __BIGRAPH_H */