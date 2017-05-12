package com.statnlp.example.linear_ne;

import java.util.ArrayList;
import java.util.Arrays;

import com.statnlp.commons.types.Instance;
import com.statnlp.hybridnetworks.LocalNetworkParam;
import com.statnlp.hybridnetworks.Network;
import com.statnlp.hybridnetworks.NetworkCompiler;
import com.statnlp.hybridnetworks.NetworkIDMapper;

public class ECRFNetworkCompiler extends NetworkCompiler{

	private static final long serialVersionUID = -2388666010977956073L;

	public enum NODE_TYPES {LEAF,NODE,ROOT};
	public int _size;
	public ECRFNetwork genericUnlabeledNetwork;
	
	public ECRFNetworkCompiler(){
		this._size = 150;
		this.compileUnlabeledInstancesGeneric();
	}
	
	@Override
	public ECRFNetwork compile(int networkId, Instance inst, LocalNetworkParam param) {
		// TODO Auto-generated method stub
		ECRFInstance lcrfInstance = (ECRFInstance)inst;
		if(lcrfInstance.isLabeled())
			return compileLabeledInstances(networkId, lcrfInstance, param);
		else
			return compileUnlabeledInstances(networkId, lcrfInstance, param);
	}
	
	public long toNode_leaf(){
		int[] arr = new int[]{0,Entity.get("O").getId(), 0,0,NODE_TYPES.LEAF.ordinal()};
		return NetworkIDMapper.toHybridNodeID(arr);
	}
	
	public long toNode(int pos, int tag_id){
		int[] arr = new int[]{pos+1,tag_id,0,0,NODE_TYPES.NODE.ordinal()};
		return NetworkIDMapper.toHybridNodeID(arr);
	}
	
	public long toNode_root(int size){
		int[] arr = new int[]{size+1,Entity.Entities.size(),0,0,NODE_TYPES.ROOT.ordinal()};
		return NetworkIDMapper.toHybridNodeID(arr);
	}

	@Override
	public ECRFInstance decompile(Network network) {
		ECRFNetwork lcrfNetwork = (ECRFNetwork)network;
		ECRFInstance lcrfInstance = (ECRFInstance)lcrfNetwork.getInstance();
		ECRFInstance result = lcrfInstance.duplicate();
		ArrayList<String> prediction = new ArrayList<String>();
		
		
		long root = toNode_root(lcrfInstance.size());
		int rootIdx = Arrays.binarySearch(lcrfNetwork.getAllNodes(),root);
		//System.err.println(rootIdx+" final score:"+network.getMax(rootIdx));
		for(int i=0;i<lcrfInstance.size();i++){
			int child_k = lcrfNetwork.getMaxPath(rootIdx)[0];
			long child = lcrfNetwork.getNode(child_k);
			rootIdx = child_k;
			int tagID = NetworkIDMapper.toHybridNodeArray(child)[1];
			String resEntity = Entity.get(tagID).getForm();
			if(resEntity.startsWith("S-")) resEntity = "B-"+resEntity.substring(2);
			if(resEntity.startsWith("E-")) resEntity = "I-"+resEntity.substring(2);
			prediction.add(0, resEntity);
		}
		
		result.setPrediction(prediction);
		result.setPredictionScore(lcrfNetwork.getMax());
		return result;
	}
	

	public ECRFNetwork compileLabeledInstances(int networkId, ECRFInstance inst, LocalNetworkParam param){
		ECRFNetwork lcrfNetwork = new ECRFNetwork(networkId, inst,param, this);
		long leaf = toNode_leaf();
		long[] children = new long[]{leaf};
		lcrfNetwork.addNode(leaf);
		for(int i=0;i<inst.size();i++){
			long node = toNode(i, Entity.get(inst.getOutput().get(i)).getId()   );
			lcrfNetwork.addNode(node);
			long[] currentNodes = new long[]{node};
			lcrfNetwork.addEdge(node, children);
			children = currentNodes;
		}
		long root = toNode_root(inst.size());
		lcrfNetwork.addNode(root);
		lcrfNetwork.addEdge(root, children);
		lcrfNetwork.finalizeNetwork();
		if(!genericUnlabeledNetwork.contains(lcrfNetwork))
			System.err.println("not contains");
		return lcrfNetwork;
	}
	
	public ECRFNetwork compileUnlabeledInstances(int networkId, ECRFInstance inst, LocalNetworkParam param){
		
		long[] allNodes = genericUnlabeledNetwork.getAllNodes();
		long root = toNode_root(inst.size());
		int rootIdx = Arrays.binarySearch(allNodes, root);
		ECRFNetwork lcrfNetwork = new ECRFNetwork(networkId, inst,allNodes,genericUnlabeledNetwork.getAllChildren() , param, rootIdx+1, this);
		return lcrfNetwork;
	}
	
	public void compileUnlabeledInstancesGeneric(){
		ECRFNetwork lcrfNetwork = new ECRFNetwork();
		
		long leaf = toNode_leaf();
		long[] children = new long[]{leaf};
		lcrfNetwork.addNode(leaf);
		for(int i=0;i<_size;i++){
			long[] currentNodes = new long[Entity.Entities.size()];
			for(int l=0;l<Entity.Entities.size();l++){
//				if(i==0 && Entity.get(l).getForm().startsWith("I-")){ currentNodes[l]=-1; continue;}
				long node = toNode(i,l);
				
				String currEntity = Entity.get(l).getForm();
				for(long child: children){
					if(child==-1) continue;
					int[] childArr = NetworkIDMapper.toHybridNodeArray(child);
					String childEntity = Entity.get(childArr[1]).getForm();
					
					
					if( (childEntity.startsWith("B-") || childEntity.startsWith("I-")  ) 
							&& (currEntity.startsWith("I-") || currEntity.startsWith("E-"))
							&& childEntity.substring(2).equals(currEntity.substring(2)) ) {
						if(lcrfNetwork.contains(child)){
							lcrfNetwork.addNode(node);
							lcrfNetwork.addEdge(node, new long[]{child});
						}
						
					}else if(   (childEntity.startsWith("S-") || childEntity.startsWith("E-") || childEntity.equals("O") ) 
							&& (currEntity.startsWith("B-") ||currEntity.startsWith("S-") || currEntity.startsWith("O") ) ) {
						
						if(lcrfNetwork.contains(child)){
							lcrfNetwork.addNode(node);
							lcrfNetwork.addEdge(node, new long[]{child});
						}
					}
				}
				if(lcrfNetwork.contains(node))
					currentNodes[l] = node;
				else currentNodes[l] = -1;
			}
			long root = toNode_root(i+1);
			lcrfNetwork.addNode(root);
			for(long child:currentNodes){
				if(child==-1) continue;
				lcrfNetwork.addEdge(root, new long[]{child});
			}
				
			children = currentNodes;
			
		}
		lcrfNetwork.finalizeNetwork();
		genericUnlabeledNetwork =  lcrfNetwork;
	}
	
	public double costAt(Network network, int parent_k, int[] child_k){
		return super.costAt(network, parent_k, child_k);
	}
	
}
