package com.statnlp.example;

import java.io.BufferedReader;
import java.io.File;
import java.io.FileInputStream;
import java.io.FileNotFoundException;
import java.io.FileOutputStream;
import java.io.IOException;
import java.io.InputStreamReader;
import java.io.ObjectInputStream;
import java.io.ObjectOutputStream;
import java.lang.reflect.Field;
import java.util.ArrayList;
import java.util.Collections;
import java.util.List;

import com.statnlp.commons.types.Instance;
import com.statnlp.commons.types.Label;
import com.statnlp.example.weak_semi_crf.Span;
import com.statnlp.example.weak_semi_crf.WeakSemiCRFFeatureManager;
import com.statnlp.example.weak_semi_crf.WeakSemiCRFInstance;
import com.statnlp.example.weak_semi_crf.WeakSemiCRFNetworkCompiler;
import com.statnlp.example.weak_semi_crf.WeakSemiCRFViewer;
import com.statnlp.hybridnetworks.DiscriminativeNetworkModel;
import com.statnlp.hybridnetworks.GenerativeNetworkModel;
import com.statnlp.hybridnetworks.GlobalNetworkParam;
import com.statnlp.hybridnetworks.NetworkConfig;
import com.statnlp.hybridnetworks.NetworkModel;

public class WeakSemiCRFMain {
	
	public static boolean COMBINE_OUTSIDE_CHARS = true;
	public static boolean USE_SINGLE_OUTSIDE_TAG = true;
	
	private static GlobalNetworkParam param;
	
	public static void main(String[] args) throws FileNotFoundException, IOException, ClassNotFoundException, NoSuchFieldException, SecurityException, InterruptedException, IllegalArgumentException, IllegalAccessException{
		boolean serializeModel = false;
		boolean useCoNLLData = false;
		boolean limitNumInstances = true;
		boolean visualize = true;
		
		String train_filename;
		String test_filename;
		WeakSemiCRFInstance[] trainInstances; 
		WeakSemiCRFInstance[] testInstances;
		
		if(useCoNLLData){
			train_filename = "data/SMSNP/SMSNP.conll.train";
			test_filename = "data/SMSNP/SMSNP.conll.test";
			trainInstances = readCoNLLData(train_filename, true);
			testInstances = readCoNLLData(test_filename, false);
		} else {
			train_filename = "data/SMSNP/SMSNP.train";
			test_filename = "data/SMSNP/SMSNP.test";
			trainInstances = readData(train_filename, true);
			testInstances = readData(test_filename, false);
		}
		if(limitNumInstances){
			int limit = 100;
			WeakSemiCRFInstance[] tmp = new WeakSemiCRFInstance[limit];
			for(int i=0; i<limit; i++){
				tmp[i] = trainInstances[i];
			}
			trainInstances = tmp;
			tmp = new WeakSemiCRFInstance[limit];
			for(int i=0; i<limit; i++){
				tmp[i] = testInstances[i];
			}
			testInstances = tmp;
		}
		
		int maxSize = 0;
		int maxSpan = 0;
		for(WeakSemiCRFInstance instance: trainInstances){
			maxSize = Math.max(maxSize, instance.size());
			for(Span span: instance.output){
				maxSpan = Math.max(maxSpan, span.end-span.start);
			}
		}
		for(WeakSemiCRFInstance instance: testInstances){
			maxSize = Math.max(maxSize, instance.size());
		}
		
		NetworkConfig.TRAIN_MODE_IS_GENERATIVE = false;
		NetworkConfig.CACHE_FEATURES_DURING_TRAINING = true;
		NetworkConfig.L2_REGULARIZATION_CONSTANT = 0.01;
		NetworkConfig.OBJTOL = 1e-2;
		NetworkConfig.NUM_THREADS = 4;
		NetworkConfig.PARALLEL_FEATURE_EXTRACTION = true;
		
		int numIterations = 5000;
		
		int size = trainInstances.length;
		
		System.err.println("Read.."+size+" instances.");
		
		param = new GlobalNetworkParam();
		
		WeakSemiCRFFeatureManager fm = new WeakSemiCRFFeatureManager(param);
		
		Label[] labels = param.LABELS.values().toArray(new Label[param.LABELS.size()]);
		
		WeakSemiCRFNetworkCompiler compiler = new WeakSemiCRFNetworkCompiler(labels, maxSize, maxSpan);
		
		NetworkModel model = NetworkConfig.TRAIN_MODE_IS_GENERATIVE ? GenerativeNetworkModel.create(fm, compiler) : DiscriminativeNetworkModel.create(fm, compiler);
		if(visualize){
			model.visualize(WeakSemiCRFViewer.class, trainInstances);
		}
		
		if(serializeModel){
			String modelPath = "experiments/SMSNP/SMSNP.allFeatures.alldata.model";
			if(new File(modelPath).exists()){
				System.out.println("Reading object...");
				long startTime = System.currentTimeMillis();
				ObjectInputStream ois = new ObjectInputStream(new FileInputStream(modelPath));
				model = (NetworkModel)ois.readObject();
				ois.close();
				Field _fm = NetworkModel.class.getDeclaredField("_fm");
				_fm.setAccessible(true);
				fm = (WeakSemiCRFFeatureManager)_fm.get(model);
				long endTime = System.currentTimeMillis();
				System.out.printf("Done in %.3fs\n", (endTime-startTime)/1000.0);
			} else {
				model.train(trainInstances, numIterations);
				System.out.println("Writing object...");
				long startTime = System.currentTimeMillis();
				ObjectOutputStream oos = new ObjectOutputStream(new FileOutputStream(modelPath));
				oos.writeObject(model);
				oos.close();
				long endTime = System.currentTimeMillis();
				System.out.printf("Done in %.3fs\n", (endTime-startTime)/1000.0);
			}
		} else {
			model.train(trainInstances, numIterations);
		}
		
		int k = 200;
		Instance[] predictions = model.decode(testInstances, k);
		int corr = 0;
		int totalGold = 0;
		int totalPred = 0;
		for(Instance inst: predictions){
			WeakSemiCRFInstance instance = (WeakSemiCRFInstance)inst;
			System.out.println("Input:");
			System.out.println(instance.input);
			System.out.println("Gold:");
			System.out.println(instance.output);
			System.out.println("Prediction:");
			System.out.println(instance.prediction);
			List<Span> goldSpans = instance.output;
			List<Span> predSpans = instance.prediction;
			int curTotalGold = goldSpans.size();
			totalGold += curTotalGold;
			int curTotalPred = predSpans.size();
			totalPred += curTotalPred;
			int curCorr = countOverlaps(goldSpans, predSpans);
			corr += curCorr;
			double precision = 100.0*curCorr/curTotalPred;
			double recall = 100.0*curCorr/curTotalGold;
			double f1 = 2/((1/precision)+(1/recall));
			if(curTotalPred == 0) precision = 0.0;
			if(curTotalGold == 0) recall = 0.0;
			if(curTotalPred == 0 || curTotalGold == 0) f1 = 0.0;
			System.out.println(String.format("Correct: %1$3d, Predicted: %2$3d, Gold: %3$3d ", curCorr, curTotalPred, curTotalGold));
			System.out.println(String.format("Overall P: %#5.2f%%, R: %#5.2f%%, F: %#5.2f%%", precision, recall, f1));
			System.out.println();
			printScore(new Instance[]{instance});
			System.out.println();
		}
		System.out.println();
		System.out.println("### Overall score ###");
		System.out.println(String.format("Correct: %1$3d, Predicted: %2$3d, Gold: %3$3d ", corr, totalPred, totalGold));
		double precision = 100.0*corr/totalPred;
		double recall = 100.0*corr/totalGold;
		double f1 = 2/((1/precision)+(1/recall));
		if(totalPred == 0) precision = 0.0;
		if(totalGold == 0) recall = 0.0;
		if(totalPred == 0 || totalGold == 0) f1 = 0.0;
		System.out.println(String.format("Overall P: %#5.2f%%, R: %#5.2f%%, F: %#5.2f%%", precision, recall, f1));
		System.out.println();
		printScore(predictions);
	}
	
	private static List<Span> duplicate(List<Span> list){
		List<Span> result = new ArrayList<Span>();
		for(Span span: list){
			result.add(span);
		}
		return result;
	}
	
	private static void printScore(Instance[] instances){
		int size = param.LABELS.size();
		int[] corrects = new int[size];
		int[] totalGold = new int[size];
		int[] totalPred = new int[size];
		for(Instance inst: instances){
			WeakSemiCRFInstance instance = (WeakSemiCRFInstance)inst;
			List<Span> predicted = duplicate(instance.getPrediction());
			List<Span> actual = duplicate(instance.getOutput());
			for(Span span: actual){
				if(predicted.contains(span)){
					predicted.remove(span);
					Label label = span.label;
					corrects[label.getId()] += 1;
					totalPred[label.getId()] += 1;
				}
				totalGold[span.label.getId()] += 1;
			}
			for(Span span: predicted){
				totalPred[span.label.getId()] += 1;
			}
		}
		double avgF1 = 0;
		for(int i=0; i<size; i++){
			double precision = (totalPred[i] == 0) ? 0.0 : 1.0*corrects[i]/totalPred[i];
			double recall = (totalGold[i] == 0) ? 0.0 : 1.0*corrects[i]/totalGold[i];
			double f1 = (precision == 0.0 || recall == 0.0) ? 0.0 : 2/((1/precision)+(1/recall));
			avgF1 += f1;
			System.out.println(String.format("%6s: #Corr:%2$3d, #Pred:%3$3d, #Gold:%4$3d, Pr=%5$#5.2f%% Rc=%6$#5.2f%% F1=%7$#5.2f%%", param.getLabel(i).getForm(), corrects[i], totalPred[i], totalGold[i], precision*100, recall*100, f1*100));
		}
		System.out.printf("Macro average F1: %.2f%%", 100*avgF1/size);
	}
	
	/**
	 * Count the number of overlaps (common elements) in the given lists.
	 * Duplicate objects are counted as separate objects.
	 * @param list1
	 * @param list2
	 * @return
	 */
	private static int countOverlaps(List<Span> list1, List<Span> list2){
		int result = 0;
		List<Span> copy = new ArrayList<Span>();
		copy.addAll(list2);
		for(Span span: list1){
			if(copy.contains(span)){
				copy.remove(span);
				result += 1;
			}
		}
		return result;
	}
	
	/**
	 * Read data from a file with three-line format:<br>
	 * - First line the input string<br>
	 * - Second line the list of spans in the format "start,end Label" separated by pipe "|"<br>
	 * - Third line an empty line
	 * @param fileName
	 * @param isLabeled
	 * @return
	 * @throws IOException
	 */
	private static WeakSemiCRFInstance[] readData(String fileName, boolean isLabeled) throws IOException{
		InputStreamReader isr = new InputStreamReader(new FileInputStream(fileName), "UTF-8");
		BufferedReader br = new BufferedReader(isr);
		ArrayList<WeakSemiCRFInstance> result = new ArrayList<WeakSemiCRFInstance>();
		String input = null;
		List<Span> output = null;
		int instanceId = 1;
		while(br.ready()){
			input = br.readLine();
			int length = input.length();
			output = new ArrayList<Span>();
			String[] spansStr = br.readLine().split("\\|");
			List<Span> spans = new ArrayList<Span>();
			for(String span: spansStr){
				if(span.length() == 0){
					continue;
				}
				String[] startend_label = span.split(" ");
				Label label = param.getLabel(startend_label[1]);
				String[] start_end = startend_label[0].split(",");
				int start = Integer.parseInt(start_end[0]);
				int end = Integer.parseInt(start_end[1]);
				spans.add(new Span(start, end, label));
			}
			Collections.sort(spans); // Ensure it is sorted
			
			int prevEnd = 0;
			for(Span span: spans){
				int start = span.start;
				int end = span.end;
				Label label = span.label;
				if(prevEnd < start){
					createOutsideSpans(input, output, prevEnd, start);
				}
				prevEnd = end;
				output.add(new Span(start, end, label));
			}
			createOutsideSpans(input, output, prevEnd, length);
			WeakSemiCRFInstance instance = new WeakSemiCRFInstance(instanceId, 1.0, input, output);
			if(isLabeled){
				instance.setLabeled();
			} else {
				instance.setUnlabeled();
			}
			result.add(instance);
			instanceId += 1;
			br.readLine();
		}
		br.close();
		return result.toArray(new WeakSemiCRFInstance[result.size()]);
	}
	
	/**
	 * Create the outside spans in the specified substring
	 * @param input
	 * @param output
	 * @param start
	 * @param end
	 */
	private static void createOutsideSpans(String input, List<Span> output, int start, int end){
		int length = input.length();
		int curStart = start;
		while(curStart < end){
			int curEnd = input.indexOf(' ', curStart);
			Label outsideLabel = null;
			if(USE_SINGLE_OUTSIDE_TAG){
				outsideLabel = param.getLabel("O");
				if(curEnd == -1 || curEnd > end){
					curEnd = end;
				} else if(curStart == curEnd){
					curEnd += 1;
				}
			} else {
				if(curEnd == -1 || curEnd > end){ // No space
					curEnd = end;
					if(curStart == start){ // Start directly after previous tag: this is between tags
						if(curStart == 0){ // Unless this is the start of the string
							if(curEnd == length){
								outsideLabel = param.getLabel("O"); // Case |<cur>|
							} else {
								outsideLabel = param.getLabel("O-B"); // Case |<cur>###
							}
						} else {
							if(curEnd == length){
								outsideLabel = param.getLabel("O-A"); // Case ###<cur>|
							} else {
								outsideLabel = param.getLabel("O-I"); // Case ###<cur>###
							}
						}
					} else { // Start not immediately: this is before tags (found space before)
						if(curEnd == length){
							outsideLabel = param.getLabel("O"); // Case ### <cur>|
						} else {
							outsideLabel = param.getLabel("O-B"); // Case ### <cur>###
						}
					}
				} else if(curStart == curEnd){ // It is immediately a space
					curEnd += 1;
					outsideLabel = param.getLabel("O"); // Tag space as a single outside token
				} else if(curStart < curEnd){ // Found a non-immediate space
					if(curStart == start){ // Start immediately after previous tag: this is after tag
						if(curStart == 0){
							outsideLabel = param.getLabel("O"); // Case |<cur> ###
						} else {
							outsideLabel = param.getLabel("O-A"); // Case ###<cur> ###
						}
					} else { // Start not immediately: this is a separate outside token
						outsideLabel = param.getLabel("O"); // Case ### <cur> ###
					}
				}
			}
			output.add(new Span(curStart, curEnd, outsideLabel));
			curStart = curEnd;
		}
	}
	
	/**
	 * Read data from file in a CoNLL format 
	 * @param fileName
	 * @param isLabeled
	 * @return
	 * @throws IOException
	 */
	private static WeakSemiCRFInstance[] readCoNLLData(String fileName, boolean isLabeled) throws IOException{
		InputStreamReader isr = new InputStreamReader(new FileInputStream(fileName), "UTF-8");
		BufferedReader br = new BufferedReader(isr);
		ArrayList<WeakSemiCRFInstance> result = new ArrayList<WeakSemiCRFInstance>();
		String input = null;
		List<Span> output = null;
		int instanceId = 1;
		int start = -1;
		int end = 0;
		Label prevLabel = null;
		while(br.ready()){
			if(input == null){
				input = "";
				output = new ArrayList<Span>();
				start = -1;
				end = 0;
				prevLabel = null;
			}
			String line = br.readLine().trim();
			if(line.length() == 0){
				input = input.trim();
				end = input.length();
				if(start != -1){
					createSpan(output, start, end, prevLabel);
				}
				WeakSemiCRFInstance instance = new WeakSemiCRFInstance(instanceId, 1);
				instance.input = input;
				instance.output = output;
				if(isLabeled){
					instance.setLabeled(); // Important!
				} else {
					instance.setUnlabeled();
				}
				instanceId++;
				result.add(instance);
				input = null;
			} else {
				int lastSpace = line.lastIndexOf(" ");
				String word = line.substring(0, lastSpace);
				String form = line.substring(lastSpace+1);
				Label label = null;
				end = input.length();
				if(form.startsWith("B")){
					if(start != -1){
						createSpan(output, start, end, prevLabel);
					}
					if(prevLabel != null && !prevLabel.getForm().matches("O-[BI]")){
						// Assumption: consecutive non-outside tags are separated by a space
						input += " ";
						createSpan(output, end, end+1, param.getLabel("O"));
						end += 1;
					}
					start = end;
					input += word;
					label = param.getLabel(form.substring(form.indexOf("-")+1));
				} else if(form.startsWith("I")){
					input += " "+word;
					label = param.getLabel(form.substring(form.indexOf("-")+1));
				} else if(form.startsWith("O")){
					if(start != -1){
						createSpan(output, start, end, prevLabel);
					}
					if(prevLabel != null && form.matches("O(-B)?")){
						input += " ";
						createSpan(output, end, end+1, param.getLabel("O"));
						end += 1;
					}
					start = end;
					input += word;
					if(USE_SINGLE_OUTSIDE_TAG){
						label = param.getLabel("O");
					} else {
						label = param.getLabel(form);
					}
				}
				prevLabel = label;
			}
		}
		br.close();
		return result.toArray(new WeakSemiCRFInstance[result.size()]);
	}
	
	private static void createSpan(List<Span> output, int start, int end, Label label){
		if(label.getForm().startsWith("O")){
			if(COMBINE_OUTSIDE_CHARS){
				output.add(new Span(start, end, label));
			} else {
				for(int i=start; i<end; i++){
					output.add(new Span(i, i+1, label));
				}
			}
		} else {
			output.add(new Span(start, end, label));
		}
	}

}
