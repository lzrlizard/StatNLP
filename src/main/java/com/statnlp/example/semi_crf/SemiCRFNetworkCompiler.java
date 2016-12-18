package com.statnlp.example.semi_crf;

import java.util.ArrayList;
import java.util.Arrays;
import java.util.Collections;
import java.util.List;

import com.statnlp.commons.types.Instance;
import com.statnlp.hybridnetworks.LocalNetworkParam;
import com.statnlp.hybridnetworks.Network;
import com.statnlp.hybridnetworks.NetworkCompiler;
import com.statnlp.hybridnetworks.NetworkException;
import com.statnlp.hybridnetworks.NetworkIDMapper;

public class SemiCRFNetworkCompiler extends NetworkCompiler {
	
	private final static boolean DEBUG = false;
	
	private static final long serialVersionUID = 6585870230920484539L;
	public Label[] labels;
	public int maxSize = 500;
	public int maxSegmentLength = 20;
	public long[] allNodes;
	public int[][][] allChildren;
	
	public enum NodeType {
		LEAF,
		BEGIN,
		END,
		ROOT,
	}
	
	static {
		NetworkIDMapper.setCapacity(new int[]{10000, 10, 100});
	}

	public SemiCRFNetworkCompiler(Label[] labels, int maxSize, int maxSegmentLength) {
		this.labels = labels;
		this.maxSize = Math.max(maxSize, this.maxSize);
		this.maxSegmentLength = Math.max(maxSegmentLength, this.maxSegmentLength);
		System.out.println(String.format("Max size: %s, Max segment length: %s", maxSize, maxSegmentLength));
		System.out.println(Arrays.asList(labels));
		buildUnlabeled();
	}

	@Override
	public SemiCRFNetwork compile(int networkId, Instance inst, LocalNetworkParam param) {
		try{
			if(inst.isLabeled()){
				return compileLabeled(networkId, (SemiCRFInstance)inst, param);
			} else {
				return compileUnlabeled(networkId, (SemiCRFInstance)inst, param);
			}
		} catch (NetworkException e){
			System.out.println(inst);
			throw e;
		}
	}
	
	private SemiCRFNetwork compileLabeled(int networkId, SemiCRFInstance instance, LocalNetworkParam param){
		SemiCRFNetwork network = new SemiCRFNetwork(networkId, instance, param);
		
		int size = instance.size();
		List<Span> output = instance.getOutput();
		Collections.sort(output);
		
		long leaf = toNode_leaf();
		network.addNode(leaf);
		long prevNode = leaf;
		for(Span span: output){
			int labelId = span.label.id;
			long begin = toNode_begin(span.start, labelId);
			long end = toNode_end(span.end-1, labelId);
			network.addNode(begin);
			network.addNode(end);
			network.addEdge(begin, new long[]{prevNode});
			network.addEdge(end, new long[]{begin});
			prevNode = end;
		}
		long root = toNode_root(size-1);
		network.addNode(root);
		network.addEdge(root, new long[]{prevNode});
		
		network.finalizeNetwork();
		
		if(DEBUG){
			System.out.println(network);
			SemiCRFNetwork unlabeled = compileUnlabeled(networkId, instance, param);
			System.out.println("Contained: "+unlabeled.contains(network));
		}
		return network;
	}
	
	private SemiCRFNetwork compileUnlabeled(int networkId, SemiCRFInstance instance, LocalNetworkParam param){
		int size = instance.size();
		long root = toNode_root(size-1);
		int root_k = Arrays.binarySearch(allNodes, root);
		int numNodes = root_k + 1;
		return new SemiCRFNetwork(networkId, instance, allNodes, allChildren, param, numNodes);
	}
	
	private void buildUnlabeled(){
		SemiCRFNetwork network = new SemiCRFNetwork();
		
		long leaf = toNode_leaf();
		network.addNode(leaf);
		List<Long> prevNodes = new ArrayList<Long>();
		List<Long> currNodes = new ArrayList<Long>();
		prevNodes.add(leaf);
		for(int pos=0; pos<maxSize; pos++){
			for(int labelId=0; labelId<labels.length; labelId++){
				long beginNode = toNode_begin(pos, labelId);
				long endNode = toNode_end(pos, labelId);
				
				network.addNode(beginNode);
				network.addNode(endNode);
				
				currNodes.add(endNode);
				
				for(int prevPos=pos; prevPos > pos-maxSegmentLength && prevPos >= 0; prevPos--){
					long prevBeginNode = toNode_begin(prevPos, labelId);
					network.addEdge(endNode, new long[]{prevBeginNode});
				}
				
				for(long prevNode: prevNodes){
					network.addEdge(beginNode, new long[]{prevNode});
				}
			}
			long root = toNode_root(pos);
			network.addNode(root);
			for(long currNode: currNodes){
				network.addEdge(root, new long[]{currNode});	
			}
			prevNodes = currNodes;
			currNodes = new ArrayList<Long>();
		}
		network.finalizeNetwork();
		allNodes = network.getAllNodes();
		allChildren = network.getAllChildren();
	}
	
	private long toNode_leaf(){
		return toNode(0, 0, NodeType.LEAF);
	}
	
	private long toNode_begin(int pos, int labelId){
		return toNode(pos, labelId, NodeType.BEGIN);
	}
	
	private long toNode_end(int pos, int labelId){
		return toNode(pos, labelId, NodeType.END);
	}
	
	private long toNode_root(int pos){
		return toNode(pos, labels.length, NodeType.ROOT);
	}
	
	private long toNode(int pos, int labelId, NodeType type){
		return NetworkIDMapper.toHybridNodeID(new int[]{pos, type.ordinal(), labelId});
	}

	@Override
	public SemiCRFInstance decompile(Network net) {
		SemiCRFNetwork network = (SemiCRFNetwork)net;
		SemiCRFInstance result = (SemiCRFInstance)network.getInstance().duplicate();
		List<Span> prediction = new ArrayList<Span>();
		int node_k = network.countNodes()-1;
		while(node_k > 0){
			int[] children_k = network.getMaxPath(node_k);
			int[] child_arr = network.getNodeArray(children_k[0]);
			int end = child_arr[0];
			NodeType nodeType = NodeType.values()[child_arr[1]];
			if(nodeType == NodeType.LEAF){
				break;
			} else {
				assert nodeType == NodeType.END;
			}
			int labelId = child_arr[2];
			children_k = network.getMaxPath(children_k[0]);
			child_arr = network.getNodeArray(children_k[0]);
			int start = child_arr[0];
			prediction.add(new Span(start, end+1, Label.get(labelId)));
			node_k = children_k[0];
		}
		Collections.sort(prediction);
		result.setPrediction(prediction);
		return result;
	}

}
