StatNLP
=====
What’s StatNLP? </br>
------------
This package contains StatNLP Hypergraph-based Graphical Model: A Framework for
Structured Prediction, or StatNLP Framework for short.

This framework is based on the concept of hypergraph, making it very general,
which covers linear-chain graphical models -- such as HMM, Linear-chain CRFs,
and semi-Markov CRFs --, tree-based graphical models -- such as Tree CRFs for 
constituency parsers --, and many more.

Coupled with a generic training formulation based on the generalization of the
inside-outside algorithm to acyclic hypergraphs, this framework supports the 
rapid prototyping and creation of novel graphical models, where users would 
just need to specify the graphical structure, and the framework will handle the
training procedure.

This framework has been successfully used to produce novel models in the 
following research papers:

- Efficient Dependency-guided Named Entity Recognition
  Zhanming Jie, Aldrian Obaja Muis, and Wei Lu, AAAI 2017

- Learning Latent Sentiment Scopes for Entity-level Sentiment Analysis
  Hao Li and Wei Lu, AAAI 2017

- Semantic Parsing with Neural Hybrid Trees
  Raymond Hendy Susanto and Wei Lu, AAI 2017

- A General Regularization Framework for Domain Adaptation
  Wei Lu, Hai Leong Chieu, and Jonathan Löfgren, EMNLP 2016

- Learning to Recognize Discontiguous Entities
  Aldrian Obaja Muis and Wei Lu, EMNLP 2016

- Weak Semi-Markov CRFs for NP Chunking in Informal Text
  Aldrian Obaja Muis and Wei Lu, NAACL 2016

- Joint Mention Extraction and Classification with Mention Hypergraphs
  Wei Lu and Dan Roth, EMNLP 2015

- Constrained Semantic Forests for Improved Discriminative Semantic Parsing
  Wei Lu, EMNLP 2015

- Semantic Parsing with Relaxed Hybrid Trees
  Wei Lu, EMNLP 2014

(http://www.statnlp.org/)

Direct usage </br>
------------
To compile, ensure Maven is installed, and simply do:

    mvn clean package

This will create a runnable JAR in the target/ directory.
Running the JAR file directly without any parameter will show the help:

    java -jar target/statnlp-core-{VERSION}.jar

For example:

    java -jar target/statnlp-core-2017.1-SNAPSHOT.jar

The package comes with some predefined models, which you can directly use with
your data:

Linear-chain CRF:

    java -jar target/statnlp-core-2017.1-SNAPSHOT.jar \
        --linearModelClass com.statnlp.example.linear_crf.LinearCRF \
        --trainPath data/train.data \
        --testPath data/test.data \
        --modelPath data/test.model \
        train test evaluate

The last line above defines the tasks to be executed, in that order.

This package also comes with visualization GUI to see how the graphical models
represent the input. Simply execute "visualize" task to the above, as follows:

    java -jar target/statnlp-core-2017.1-SNAPSHOT.jar \
        --linearModelClass com.statnlp.example.linear_crf.LinearCRF \
        --trainPath data/train.data \
        visualize

In the visualization, you can drag the canvas to move around, and you can 
scroll to zoom in/zoom out.
Also the arrow keys (left and right) can be used to show the previous and next
instances.

Building new models </br>
------------
To help understanding the whole package, here we first describe the components
of the framework:
- FeatureManager - The class where you can implement the feature extractor
- NetworkCompiler - The class where you can implement the graphical model
- Instance - The data structure to store the input and the corresponding
             network structures. This is also used to store the gold output and
             the predictions made by the model during testing.

If your task requires input which is linear (e.g., tokenized sentences), then
you can use the built-in LinearInstance<OUT> where OUT is the output type, such
as Label in the case of POS tagging, or Tree in constituency parsing.
If the input is linear, there is also a built-in FeatureManager than can take
in feature templates: TemplateBasedFeatureManager, similar to the feature
template used in CRF++.

The main class you may want to implement is then the NetworkCompiler, which is
the core part where the graphical models are defined.

This guide will be updated in the future, but for now, you can follow the 
examples written in the src/main/java/com/statnlp/example directory, with 
the corresponding main class at src/main/test/com/statnlp/example directory to
execute the models.


Who are we? </br>
------------
Research Group in the ISTD pillar at the Singapore University of Technology and Design. Researchers in statNLP focus on solving novel research problems in the NLP, machine learning and big data that may surpass traditional boundaries of data science technology.
The statNLP was formed in 2014. Currently, it is led by Prof Lu Wei, consisting of several graduate students and researcher. The group subscribes to the design spirit of SUTD.
Nevertheless, in a short period of time since its inception, the group has built a successful track record of publishing in the top ranked conferences. Members of statNLP also collaborate with external organization such DSO National Laboratory.

Licensing </br>
---------
Statistical Natural Language Processing System </br>
Copyright (C) 2014-2016  Lu, Wei </br>
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
  
Requirements
------------
StatNLP Framework (JRE required)

Contacts
--------
joelowj@gmail.com
