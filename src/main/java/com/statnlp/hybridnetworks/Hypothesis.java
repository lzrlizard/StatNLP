/**
 * 
 */
package com.statnlp.hybridnetworks;

import java.util.ArrayList;

/**
 * This class represents a (possibly partial) hypothesis of
 * an output structure.
 * There are two types of Hypothesis: {@link NodeHypothesis} and {@link EdgeHypothesis}
 */
public abstract class Hypothesis {

	/**
	 * The node index in which this hypothesis is applicable
	 * For NodeHypothesis, this represents that node's index.
	 * For EdgeHypothesis, this represents the node index of the parent node.
	 */
	protected int nodeIndex;
	/**
	 * The children of this hypothesis, which is the previous partial hypothesis.
	 * For EdgeHypothesis, the children would be a list of NodeHypothesis.
	 * Similarly, for NodeHypothesis, the children would be a list of EdgeHypothesis.
	 */
	protected Hypothesis[] children;
	/**
	 * Whether there are more hypothesis to be predicted.
	 * Note that this is different from simply checking whether the queue is empty,
	 * because the queue is populated only when necessary by looking at the last best index.
	 */
	protected boolean hasMoreHypothesis;
	/**
	 * The priority queue storing the possible next best child.
	 * Since this is a priority queue, the next best child is the one in front of the queue.
	 */
	protected BoundedPrioritySet<ScoredIndex> nextBestChildQueue;
	/**
	 * The cache to store the list of best children, which will contain the list of 
	 * best children up to the highest k on which {@link #getKthBestHypothesis(int)} has been called.
	 */
	protected ArrayList<ScoredIndex> bestChildrenList;

	protected void init() {
		nextBestChildQueue = new BoundedPrioritySet<ScoredIndex>();
		bestChildrenList = new ArrayList<ScoredIndex>();
		hasMoreHypothesis = true;
	}
	
	/**
	 * Returns the k-th best path at this hypothesis.
	 * @param k
	 * @return
	 */
	public ScoredIndex getKthBestHypothesis(int k){
		// Assuming the k is 0-based. So k=0 will return the best prediction
		// Below we fill the cache until we satisfy the number of top-k paths requested.
		while(bestChildrenList.size() <= k){
			ScoredIndex nextBest = setAndReturnNextBestPath();
			if(nextBest == null){
				return null;
			}
		}
		return bestChildrenList.get(k);
	}
	
	/**
	 * Return the next best path, or return null if there is no next best path.
	 * @return
	 */
	public abstract ScoredIndex setAndReturnNextBestPath();
	
	public int nodeIndex(){
		return this.nodeIndex;
	}
	
	/**
	 * Sets node index of this hypothesis accordingly.
	 * For NodeHypothesis, this should be that node's index.
	 * For EdgeHypothesis, this should be the node index of the parent node.
	 * @param nodeIndex
	 */
	public void setNodeIndex(int nodeIndex){
		this.nodeIndex = nodeIndex;
	}
	
	/**
	 * @return The children of this hypothesis.
	 */
	public Hypothesis[] children() {
		return children;
	}
	
	/**
	 * @param children The children to set
	 */
	public void setChildren(Hypothesis[] children) {
		this.children = children;
	}
	
	/**
	 * Returns the last best index calculated on this hypothesis.
	 * @return
	 */
	public ScoredIndex getLastBestIndex(){
		if(bestChildrenList.size() == 0){
			return null;
		}
		return bestChildrenList.get(bestChildrenList.size()-1);
	}

	public ArrayList<ScoredIndex> bestChildrenList() {
		return bestChildrenList;
	}

	public void setBestChildrenList(ArrayList<ScoredIndex> bestChildrenList) {
		this.bestChildrenList = bestChildrenList;
	}

	public BoundedPrioritySet<ScoredIndex> nextBestChildQueue() {
		return nextBestChildQueue;
	}

	public void setNextBestChildQueue(BoundedPrioritySet<ScoredIndex> nextBestChildQueue) {
		this.nextBestChildQueue = nextBestChildQueue;
	}

}
