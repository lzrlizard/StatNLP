/** Statistical Natural Language Processing System
    Copyright (C) 2014-2016  Lu, Wei

    This program is free software: you can redistribute it and/or modify
    it under the terms of the GNU General Public License as published by
    the Free Software Foundation, either version 3 of the License, or
    (at your option) any later version.

    This program is distributed in the hope that it will be useful,
    but WITHOUT ANY WARRANTY; without even the implied warranty of
    MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
    GNU General Public License for more details.

    You should have received a copy of the GNU General Public License
    along with this program.  If not, see <http://www.gnu.org/licenses/>.
 */
package com.statnlp.hybridnetworks;

import java.io.Serializable;
import java.util.Arrays;

/**
 * The class storing a list of features by their indices.<br>
 * The instances of this class may be chained together to form a sequence of feature arrays.<br>
 * The advantage of chaining becomes apparent when feature score caching (note that this is different
 * from feature caching, where the feature indices themselves are cached) is used, since if only the weights
 * of some features change, only those feature arrays with changes will be recalculated.<br>
 * In the implementation, the feature indices are stored inside another class: {@link FeatureBox}.
 */
public class FeatureArray implements Serializable{

	private static final long serialVersionUID = 9170537017171193020L;

	/** The total score of this feature array, including all other feature arrays which are chained after this. */
	private double _totalScore;
	/** The internal feature box containing the feature indices and the cached local score. */
	private FeatureBox _fb;
	/** The flag signifying the scope of the feature indices, whether local (per thread) or global (combined feature map). */
	protected boolean _isLocal = false;

	/** An empty feature array */
	public static final FeatureArray EMPTY = new FeatureArray(new int[0]);
	/** A feature with very negative score, used to signify that the hyperpath containing this feature should not be selected */
	public static final FeatureArray NEGATIVE_INFINITY = new FeatureArray(Double.NEGATIVE_INFINITY);

	private FeatureArray _next;
	protected int[] dstNodes;
	
	/**
	 * Merges the features in <code>fs</code> and in <code>next</code><br>
	 * <strong>IMPORTANT NOTE:</strong> to use caching, please use {@link FeatureManager#createFeatureArray(Network, int[], FeatureArray)} instead,
	 * combined with setting {@link NetworkConfig#AVOID_DUPLICATE_FEATURES} to true.
	 * @param fs
	 * @param next
	 */
	public FeatureArray(int[] fs, FeatureArray next) {
		this._fb = new FeatureBox(fs);
		this._next = next;
	}

	/**
	 * Construct a feature array containing the features identified by their indices<br>
	 * <strong>IMPORTANT NOTE:</strong> to use caching, please use {@link FeatureManager#createFeatureArray(Network, int[])} instead,
	 * combined with setting {@link NetworkConfig#AVOID_DUPLICATE_FEATURES} to true.
	 * @param fs
	 */
	public FeatureArray(int[] fs) {
		this._fb = new FeatureBox(fs);
		this._next = null;
		this._isLocal = false;
	}

	/**
	 * Creates a new FeatureArray based on the feature indices in the given FeatureBox object.<br>
	 * If you do not have FeatureBox object, perhaps you might want to check {@link #FeatureArray(int[])}.
	 * @param fb
	 * @see #FeatureArray(int[])
	 * @see #FeatureArray(int[], FeatureArray)
	 */
	public FeatureArray(FeatureBox fb) {
		this(fb, null);
		this._isLocal = false;
	}

	/**
	 * Creates a new FeatureArray based on the feature indices in the given FeatureBox object.<br>
	 * If you do not have FeatureBox object, perhaps you might want to check {@link #FeatureArray(int[])}.
	 * @param fb
	 * @param next
	 * @see #FeatureArray(int[])
	 * @see #FeatureArray(int[], FeatureArray)
	 */
	public FeatureArray(FeatureBox fb, FeatureArray next) {
		this._fb = fb;
		this._next = next;
		this._isLocal = false;
	}

	private FeatureArray(double score) {
		this._totalScore = score;
	}

	public void setAlwaysChange(boolean alwaysChange){
		this._fb._alwaysChange = alwaysChange;
	}
	
	public void setDstNodes(int[] dstNodes) {
		this.dstNodes = dstNodes;
	}
	
	public FeatureArray toLocal(LocalNetworkParam param){
		if(this==NEGATIVE_INFINITY){
			return this;
		}
		if(this._isLocal){
			return this;
		}

		int length = this._fb.length();
		if(NetworkConfig.BUILD_FEATURES_FROM_LABELED_ONLY){
			for(int fs: this._fb.get()){
				if(fs == -1){
					length--;
				}
			}
		}

		int[] fs_local = new int[length];
		int localIdx = 0;
		for(int k = 0; k<this._fb.length(); k++, localIdx++){
			if(this._fb.get(k) == -1 && NetworkConfig.BUILD_FEATURES_FROM_LABELED_ONLY){
				localIdx--;
				continue;
			}
			if(!NetworkConfig.PARALLEL_FEATURE_EXTRACTION || NetworkConfig.NUM_THREADS == 1 || param._isFinalized){
				fs_local[localIdx] = param.toLocalFeature(this._fb.get(k));
			} else {
				fs_local[localIdx] = this._fb.get(k);
			}
			if(fs_local[localIdx]==-1){
				throw new RuntimeException("The local feature got an id of -1 for " + this._fb.get(k));
			}
		}

		FeatureArray fa;
		if (this._next != null){
			fa = new FeatureArray(FeatureBox.getFeatureBox(fs_local, param), this._next.toLocal(param)); //saving memory
		} else {
			fa = new FeatureArray(FeatureBox.getFeatureBox(fs_local, param)); //saving memory
		}
		fa._isLocal = true;
		fa._fb._alwaysChange = this._fb._alwaysChange;
		fa.dstNodes = this.dstNodes;
		return fa;
	}

	/**
	 * Returns the list of feature indices contained in this feature array.
	 * Note that this excludes the feature indices contained in the chained feature arrays.
	 * @return
	 */
	public int[] getCurrent(){
		return this._fb.get();
	}

	/**
	 * Sets the next feature array in this chain.
	 * @param next The next feature array in this chain.
	 */
	public void next(FeatureArray next){
		this._next = next;
	}

	/**
	 * Return the next feature array in this chain.
	 * @return The next feature array in this chain.
	 */
	public FeatureArray getNext(){
		return this._next;
	}

	public void update(LocalNetworkParam param, double count){
		if(this == NEGATIVE_INFINITY){
			return;
		}

		int[] fs_local = this.getCurrent();
		for(int f_local : fs_local){
			param.addCount(f_local, count);
		}
		
		// Recursively update the next chain
		if(this._next != null){
			this._next.update(param, count);
		}
	}
	
	
	public void update_MF_Version(LocalNetworkParam param, double count, double[] marginal){
		if(this == NEGATIVE_INFINITY){
			return;
		}

		int[] fs_local = this.getCurrent();
		for (int idx = 0; idx < fs_local.length; idx++) {
			int f_local = fs_local[idx];
			double featureValue = 1.0;
			if (this.dstNodes != null) {
				featureValue = marginal[dstNodes[idx]];
			}
			param.addCount(f_local, featureValue * count);
		}
		if(this._next != null){
			this._next.update_MF_Version(param, count, marginal);
		}
	}

	/**
	 * Return the sum of weights of the features in this array
	 * @param param
	 * @return
	 */
	public double getScore(LocalNetworkParam param, int version){
		if(this == NEGATIVE_INFINITY){
			return this._totalScore;
		}

		if(!this._isLocal != param.isGlobalMode()) {
			System.err.println(this._next);
			throw new RuntimeException("This FeatureArray is local? "+this._isLocal+"; The param is "+param.isGlobalMode());
		}

		//if the score is negative infinity, it means disabled.
		if(this._totalScore == Double.NEGATIVE_INFINITY){
			return this._totalScore;
		}
		this._totalScore = 0.0;
		if (this._fb._version != version){
			this._fb._currScore = this.computeScore(param, this.getCurrent());
			this._fb._version = version;
		}
		this._totalScore += this._fb._currScore;
		if (this._next != null){
			this._totalScore += this._next.getScore(param, version);
		}
		return this._totalScore;
	}

	/**
	 * Compute the score using the parameter and the feature array
	 * @param param
	 * @param fs
	 * @return
	 */
	private double computeScore(LocalNetworkParam param, int[] fs){
		if(!this._isLocal != param.isGlobalMode()) {
			throw new RuntimeException("This FeatureArray is local? "+this._isLocal+"; The param is "+param.isGlobalMode());
		}

		double score = 0.0;
		for(int f : fs){
			if(f!=-1){
				score += param.getWeight(f);
			} else {
				
			}
		}
		return score;
	}

	/**
	 * Get the marginal score using the marginal score as feature value
	 * @param param
	 * @param <featureIdx, targetNode> map, the target node is the corresponding node.
	 * @param marginals score array, serve as being the feature value. 
	 * @return
	 */
	public double getScore_MF_Version(LocalNetworkParam param, double[] marginal, int version){
		if(this == NEGATIVE_INFINITY){
			return this._totalScore;
		}
		if(!this._isLocal != param.isGlobalMode()) {
			throw new RuntimeException("This FeatureArray is local? "+this._isLocal+"; The param is "+param.isGlobalMode());
		}
		//if the score is negative infinity, it means disabled.
		if(this._totalScore == Double.NEGATIVE_INFINITY){
			return this._totalScore;
		}
		this._totalScore = 0.0;
		if (this._fb._version != version || this._fb._alwaysChange){
			this._fb._currScore = 0.0;
			int[] curr = this.getCurrent();
			for (int idx = 0; idx < curr.length; idx++) {
				int f = curr[idx];
			//for(int f : this.getCurrent()){
				if(f!=-1){
					//note that in training, f is the local feature index.
					//in testing, f is the global feature index
					double featureValue = 1.0;
					if (this.dstNodes != null) {
						featureValue = marginal[dstNodes[idx]];
					}
					this._fb._currScore += param.getWeight(f) * featureValue;
				}
			}
			this._fb._version = version;
		}
		this._totalScore += this._fb._currScore;

		if (this._next != null){
			this._totalScore += this._next.getScore_MF_Version(param, marginal, version);
		}
		return this._totalScore;
	}

	/**
	 * Returns the number of elements in the feature array, including all the subsequent feature arrays
	 * in the chain.
	 * @return
	 */
	public int size(){
		int size = this._fb.length();
		if (this._next != null){
			size += this._next.size();
		}
		return size;
	}

	@Override
	public String toString(){
		StringBuilder sb = new StringBuilder();
		sb.append('[');
		for(int k = 0; k<this._fb.length(); k++){
			if(k!=0)
				sb.append(' ');
			sb.append(this._fb.get(k));
		}
		sb.append(']');
		return sb.toString();
	}

	@Override
	public int hashCode(){
		int code = Arrays.hashCode(_fb._fs);
		if (this._next != null){
			code = code ^ this._next.hashCode();
		}
		return code;
	}

	@Override
	public boolean equals(Object o){
		if(o instanceof FeatureArray){
			FeatureArray fa = (FeatureArray)o;
			for(int k = 0; k< this._fb.length(); k++){
				if(this._fb.get(k) != fa._fb.get(k)){
					return false;
				}
			}
			if(this._next == null){
				if(fa._next != null){
					return false;
				}
				return true;
			}else{
				return this._next.equals(fa._next);
			}
		}
		return false;
	}

}