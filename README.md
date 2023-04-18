# Change of Relation Extraction's Entity Domain

Relation extraction (RE) is the task of discovering entities' relations in weakly structured text. There is a lot of
applications for RE such as knowledge-base population, question answering, summarization and so on. However, despite the
increasing number of studies, there is a lack of cross-domain evaluation researches. The purpose of this work is to
explore how models can be adapted to the changing types of entities.

## Motivation

There are several ways to deal with the changing types of entyties:

1) Fine-tuning

    We can retrain our model on the new obtained data, but the main problem is to get and annotate new documents

2) Ignoring

    * Build a model that does not use any information of entities' types;
    * Or don't pay any attention to the domain shift during inference.

3) Mapping

   Another way is to build a mapping from the model's entity types to another domain ones. But there may be situations
   when it is impossible to build the unambiguous mapping (e.g. diagram below, where `PER` correspond only to `PERSON`,
   but `NUM` is `NUMBER` and `TIME` concurrently).

   In the case of the unambiguous mapping, we can try all suitable mappings, but if there are $N$ entities and $M$
   candidates for each of them, $M^N$ model runs are required.

```mermaid
flowchart LR
    subgraph a["New unknown domain"]
        direction TB
        subgraph b[" "]
            direction LR
            NUM1(["NUM"])-- Date of birth --->PER1(["PER"])
            NUM2(["NUM"])-- Age --->PER2(["PER"])
        end
    end
    subgraph c["Model's domain"]
        direction TB
        subgraph d[" "]
            direction LR
            time([TIME])-- Date of birth --->person1(["PERSON"])
            number([NUMBER])-- Age --->person2(["PERSON"])
        end
    end
    a ==> c 
```

4) Diversified training

   We are going to develop training methods that instills domain shift resistance in RE models and allows them to adapt to
   new types of entities.

## Results

<table>
  <tr>
    <th rowspan="3">Adaptation methods</th>
    <th colspan="3">Results, F1-мера</th>
  </tr>
  <tr>
    <th colspan="2">DocRED</th>
    <th colspan="1">Re-TACRED</th>
  </tr>
  <tr>
    <th>SSAN-Adapt</th>
    <th>DocUNet</th>
    <th>BERT_base</th>
  </tr>
  <tr>
    <td>Ignoring</td>
    <td>53.60</td>
    <td>-</td>
    <td>-</td>
  </tr>
  <tr>
    <td>Diversified training</td>
    <td>54.23</td>
    <td>-</td>
    <td>-</td>
  </tr>
</table>


## Class diagrams

The base classes are divided into 4 main categories:

* **_Examples' features_**:
  * Span
  * FactClass
  * AbstractFact
    * EntityFact
    * RelationFact
* **_Examples_**:
  * Document
  * PreparedDocument
  * AbstractDataset
* **_Models_**:
  * AbstractModel
  * AbstractWrapperModel
* **_Utilities_**:
  * ModelManager
  * AbstractLoader

### Examples' features

```mermaid
classDiagram
directionTB
    ModelManager "1" --> "1" AbstractModelWrapper : init, train and evaluate
    AbstractModelWrapper ..> AbstractDataset : use
    AbstractDataset "1" o-- "1..*" Document : process
    AbstractDataset "1" o-- "1..*" PreparedDocument : store
    
    AbstractModelWrapper <|-- SSANAdaptModel
    AbstractModelWrapper <|-- BertBaseline
    
    class AbstractModelWrapper{
        <<Abstract>>
    }
    
    class AbstractDataset{
        <<Abstract>>
    }
    
    class PreparedDocument{
        <<NamedTuple>>
    }
```

```mermaid
classDiagram
direction TB

   Document "1" o-- "1..*" AbstractFact
   Document "1" o-- "1..*" Span
   AbstractFact <|-- EntityFact
   AbstractFact <|-- RelationFact
   Span "1" o-- "1..*" EntityFact : is mentioned in
   AbstractFact "1" --> "1" FactClass : is a
   
   class Document{
      +doc_id: str
      +text: str
      +words: Tuple[Span]
      +sentences: Tuple[Tuple[Span]]
      +facts: Tuple[AbstractFact]
      +coreference_chains: Dict[int, Tuple[EntityFact]]
      +get_word(span: Span) str
   }
   
   class Span:::rect{
      +start_idx: int
      +end_idx: int
   }
   
   class FactClass{
      <<Enumeration>>
      ENTITY
      RELATION
   }

   class AbstractFact{
      <<Abstract>> 
      +name: str
      +type_id: str
      +fact_class: FactClass
   }
   
   class EntityFact{
      +coreference_id: int
      +mentions: FrozenSet[Span]
   }
   
   class RelationFact{
      +from_fact: EntityFact
      +to_fact: EntityFact
   }
```

## Run

### Dowload datasets

1) `bash scripts/download_datasets.sh`
2) In order to download original TACRED dataset visit [LDC TACRED webpage](https://catalog.ldc.upenn.edu/LDC2018T24).
   If you are an LDC member, the access will be free; otherwise, an access fee of $25 is needed.  In addition to the original version of 
   TACRED, we should also use the new label-corrected version of the TACRED dataset, which fixed a substantial portion of the dev/test 
   labels in the original release. For more details, see the [TACRED Revisited paper](https://arxiv.org/pdf/2004.14855.pdf) and their
   original [code base](https://github.com/DFKI-NLP/tacrev)

   After downloading and processing:
   * move tacred folder to `./etc/datasets` folder 
   * put all patched files in the `./etc/dataset/tacred/data/json` directory


### Build docker container
1) `cd path/to/project`
2) `docker build ./`
3) `docker run -it --gpus=all __image_id__ /bin/bash`

### Start training

`bash scripts/main.sh -c path/to/config -v __gpu_id__ -s __seed__ -o path/to/model/output/dir`
