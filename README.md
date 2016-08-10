[![Build Status](https://travis-ci.org/charite/hyperSMURF.svg?branch=master)](https://travis-ci.org/charite/hyperSMURF)
[![API Docs](https://img.shields.io/badge/api-master-blue.svg?style=flat)](http://charite.github.io/hyperSMURF/api/master/)
[![license](https://img.shields.io/badge/licence-GNU%20GPLv3-blue.svg)]()


# hyperSMURF

Weka implementation of hyperSMURF using EasyEnsemble and SMOTE

## Requirements

hyperSMURF requires java 8 and higher. It can be used as a [Weka](http://www.cs.waikato.ac.nz/~ml/weka/) plugin using version 3.9 or higher. To build the program it is recommended to use [Maven](https://maven.apache.org/).

## Installation

To use hyperSMURF in [Weka](http://www.cs.waikato.ac.nz/~ml/weka/) three steps are needed.

1. Compile the java classes using Maven   
2. Create the Weka plugin using Maven
3. Load the plugin into your Weka Package Explorer

### Compile the java classes

