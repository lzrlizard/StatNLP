package com.statnlp.example.mention_hypergraph;

import java.util.ArrayList;
import java.util.Arrays;
import java.util.HashMap;
import java.util.HashSet;
import java.util.List;
import java.util.Map;
import java.util.Set;

import com.statnlp.commons.types.Instance;
import com.statnlp.hybridnetworks.LocalNetworkParam;
import com.statnlp.hybridnetworks.Network;
import com.statnlp.hybridnetworks.NetworkCompiler;
import com.statnlp.hybridnetworks.NetworkException;
import com.statnlp.hybridnetworks.NetworkIDMapper;

public class MentionHypergraphNetworkCompiler extends NetworkCompiler {

	private static final long serialVersionUID = -4353692924395630953L;
	public static final boolean DEBUG = false;
	
	public Label[] labels;
	public int maxSize = 200;
	public MentionHypergraphNetwork unlabeledNetwork;
	
	public enum NodeType{
		X_NODE,
		I_NODE,
		T_NODE,
		E_NODE,
		A_NODE,
	}

	public MentionHypergraphNetworkCompiler(Label[] labels, int maxSize) {
		this.labels = labels;
		this.maxSize = Math.max(maxSize, this.maxSize);
		buildUnlabeled();
	}

	@Override
	public MentionHypergraphNetwork compile(int networkId, Instance inst, LocalNetworkParam param) {
		MentionHypergraphInstance instance = (MentionHypergraphInstance)inst;
		if(instance.isLabeled()){
			return compileLabeled(networkId, instance, param);
		} else {
			return compileUnlabeled(networkId, instance, param);
		}
	}
	
	private MentionHypergraphNetwork compileLabeled(int networkId, MentionHypergraphInstance instance, LocalNetworkParam param){
		MentionHypergraphNetwork network = new MentionHypergraphNetwork(networkId, instance, param);
		int size = instance.size();
		
		long xNode = toNode_X();
		network.addNode(xNode);
		for(Span span: instance.output){
			int labelId = span.label.id;
			long prevINode = -1;
			for(int pos=span.start; pos<span.end; pos++){
				long curINode = toNode_I(pos, size, labelId);
				if(!network.contains(curINode)){
					network.addNode(curINode);
				}
				if(prevINode == -1){
					long tNode = toNode_T(pos, size, labelId);
					if(!network.contains(tNode)){
						network.addNode(tNode);
					}
					try{
						network.addEdge(tNode, new long[]{curINode});
					} catch (NetworkException e){
						// do nothing, edge from T to I already added (two mentions with the same start index)
					}
				} else {
					try{
						network.addEdge(prevINode, new long[]{curINode});
					} catch (NetworkException e){
						// do nothing, edge from prevI to curI already added (overlapping mentions)
					}
				}
				prevINode = curINode;
			}
			try{
				network.addEdge(prevINode, new long[]{xNode});
			} catch (NetworkException e){
				// do nothing, edge from I to X already added (two mentions with the same end index)
			}
		}
		for(int pos=size-1; pos>=0; pos--){
			long[] tNodes = new long[labels.length];
			for(int idx=0; idx<labels.length; idx++){
				Label label = labels[idx];
				long iNode = toNode_I(pos, size, label.id);
				if(network.contains(iNode)){
					// Convert two edges (one to X one to next I) to a single hyperedge
					ArrayList<long[]> childrenList = network.getChildren_tmp(iNode);
					if(childrenList.size() > 1){
						childrenList.clear();
						long nextINode = toNode_I(pos+1, size, label.id);
						network.addEdge(iNode, new long[]{xNode, nextINode});
					}
				}
				long tNode = toNode_T(pos, size, label.id);
				if(!network.contains(tNode)){
					network.addNode(tNode);
					network.addEdge(tNode, new long[]{xNode});
				}
				tNodes[idx] = tNode;
			}
			long eNode = toNode_E(pos, size);
			network.addNode(eNode);
			network.addEdge(eNode, tNodes);
			long aNode = toNode_A(pos, size);
			network.addNode(aNode);
			if(pos < size-1){
				long nextANode = toNode_A(pos+1, size);
				network.addEdge(aNode, new long[]{eNode, nextANode});
			} else {
				network.addEdge(aNode, new long[]{eNode});
			}
		}
		
		network.finalizeNetwork();
		
		if(DEBUG){
			MentionHypergraphNetwork unlabeled = compileUnlabeled(networkId, instance, param);
			System.out.println("Contained: "+unlabeled.contains(network));
		}
		return network;
	}
	
//	private void printArray(String[] arr){
//		StringBuilder builder = new StringBuilder();
//		builder.append("[");
//		for(String str: arr){
//			if(builder.length() > 1) builder.append(",");
//			builder.append(str);
//		}
//		builder.append("]");
//		System.out.println(builder.toString());
//	}

	private MentionHypergraphNetwork compileUnlabeled(int networkId, MentionHypergraphInstance instance, LocalNetworkParam param){
		int size = instance.size();
		long root = toNode_A(0, size);
		long[] allNodes = unlabeledNetwork.getAllNodes();
		int[][][] allChildren = unlabeledNetwork.getAllChildren();
		int root_k  = unlabeledNetwork.getNodeIndex(root);
		int numNodes = root_k+1;
		MentionHypergraphNetwork network = new MentionHypergraphNetwork(networkId, instance, allNodes, allChildren, param, numNodes);
		return network;
	}
	
	private void buildUnlabeled(){
		System.err.print("Building generic unlabeled tree up to size "+maxSize+"...");
		long startTime = System.currentTimeMillis();
		MentionHypergraphNetwork network = new MentionHypergraphNetwork();
		int size = maxSize;
		
		long xNode = toNode_X();
		network.addNode(xNode);
		for(int pos=size-1; pos>=0; pos--){
			long[] tNodes = new long[labels.length];
			for(int idx=0; idx<labels.length; idx++){
				Label label = labels[idx];
				long iNode = toNode_I(pos, size, label.id);
				network.addNode(iNode);
				network.addEdge(iNode, new long[]{xNode});
				if(pos < size-1){
					long nextINode = toNode_I(pos+1, size, label.id);
					network.addEdge(iNode, new long[]{nextINode});
					network.addEdge(iNode, new long[]{xNode, nextINode});
				}
				long tNode = toNode_T(pos, size, label.id);
				network.addNode(tNode);
				network.addEdge(tNode, new long[]{xNode});
				network.addEdge(tNode, new long[]{iNode});
				tNodes[idx] = tNode;
			}
			long eNode = toNode_E(pos, size);
			network.addNode(eNode);
			network.addEdge(eNode, tNodes);
			long aNode = toNode_A(pos, size);
			network.addNode(aNode);
			if(pos < size-1){
				long nextANode = toNode_A(pos+1, size);
				network.addEdge(aNode, new long[]{eNode, nextANode});
			} else {
				network.addEdge(aNode, new long[]{eNode});
			}
		}
		
		network.finalizeNetwork();
		
		this.unlabeledNetwork = network;
		
		long endTime = System.currentTimeMillis();
		System.err.println(String.format("Done in %.3fs", (endTime-startTime)/1000.0));
	}
	
	private long toNode_X(){
		return toNode(0, 1, NodeType.X_NODE, 0);
	}
	
	private long toNode_I(int pos, int size, int labelId){
		return toNode(pos, size, NodeType.I_NODE, labelId);
	}
	
	private long toNode_T(int pos, int size, int labelId){
		return toNode(pos, size, NodeType.T_NODE, labelId);
	}
	
	private long toNode_E(int pos, int size){
		return toNode(pos, size, NodeType.E_NODE, 0);
	}
	
	private long toNode_A(int pos, int size){
		return toNode(pos, size, NodeType.A_NODE, 0);
	}
	
	private long toNode(int pos, int size, NodeType nodeType, int labelId){
		int[] arr = new int[]{size-pos-1, nodeType.ordinal(), labelId, 0, 0};
		return NetworkIDMapper.toHybridNodeID(arr);
	}

	@Override
	public MentionHypergraphInstance decompile(Network net) {
		MentionHypergraphNetwork network = (MentionHypergraphNetwork)net;
		MentionHypergraphInstance result = (MentionHypergraphInstance)network.getInstance().duplicate();
		int size = result.size();
		
		List<Span> prediction = new ArrayList<Span>();
		long[] nodes = network.getAllNodes();
		Set<Integer> processed = new HashSet<Integer>();
		Map<Integer, Span> unprocessed = new HashMap<Integer, Span>();
		long aNode = network.getRoot();
		int aNode_k = Arrays.binarySearch(nodes, aNode);
		for(int pos=0; pos<size; pos++){
			int[] curChildren = network.getMaxPath(aNode_k);
			int eNode_k = curChildren[0];
			int[] curTNodes = network.getMaxPath(eNode_k);
			for(int idx=0; idx<curTNodes.length; idx++){
				int tNode_k = curTNodes[idx];
				int[] tNode_arr = network.getNodeArray(tNode_k);
				int labelId = tNode_arr[2];
				Label label = Label.get(labelId);
				int[] child = network.getMaxPath(tNode_k);
				int node_k = child[0]; // either an I-node or an X-node
				int[] node_arr = network.getNodeArray(node_k);
				NodeType nodeType = NodeType.values()[node_arr[1]];
				if(nodeType == NodeType.I_NODE){ // Has a mention starting here
					int start = pos;
					int end = start;
					while(nodeType == NodeType.I_NODE){
						child = network.getMaxPath(node_k);
						if(child.length == 1){
							node_k = child[0];
						} else { // there are two children for this edge (overlapping mention)
							if(processed.contains(child[1])){
								unprocessed.remove(node_k);
								node_k = child[0];
							} else {
								unprocessed.put(node_k, new Span(start, end+1, start, end+1, label));
								node_k = child[1];
								processed.add(node_k);
							}
						}
						node_arr = network.getNodeArray(node_k);
						nodeType = NodeType.values()[node_arr[1]];
						end++;
					}
					prediction.add(new Span(start, end, start, end, label));
				}
			}
			if(curChildren.length == 2){
				aNode_k = curChildren[1];
			}
		}
		for(int node_k: unprocessed.keySet()){
			prediction.add(unprocessed.get(node_k));
		}
		result.setPrediction(prediction);
		return result;
	}

}