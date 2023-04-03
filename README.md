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
    <th colspan="4">Results, F1-мера</th>
  </tr>
  <tr>
    <th colspan="2">DocRED</th>
    <th colspan="2">TACRED</th>
  </tr>
  <tr>
    <th>SSAN-Adapt</th>
    <th>DocUNet</th>
    <th>SSAN-Adapt</th>
    <th>DocUNet</th>
  </tr>
  <tr>
    <td>Ignoring</td>
    <td>53.60</td>
    <td>-</td>
    <td>-</td>
    <td>-</td>
  </tr>
  <tr>
    <td>Diversified training</td>
    <td>54.23</td>
    <td>-</td>
    <td>-</td>
    <td>-</td>
  </tr>
</table>


## Class diagrams

The base classes are divided into 3 main categories:

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
  * TorchModel
  * AbstractModel
  * AbstractWrapperModel

And `ModelManager` class that is responsible for model training and scoring 

### Examples' features
```mermaid
classDiagram
direction TB

   AbstractFact <|-- EntityFact
   AbstractFact <|-- RelationFact
   EntityFact "1" --> "1..*" Span : is mentioned in
   AbstractFact "1" --> "1" FactClass : is a
   
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
### Examples
```mermaid
classDiagram
direction LR
   class Document{
      +doc_id: str
      +text: str
      +words: Tuple[Span]
      +sentences: Tuple[Tuple[Span]]
      +facts: Tuple[AbstractFact]
      +coreference_chains: Dict[str, Tuple[EntityFact]]
      #_build_coreference_chains(facts)
      #_validate_span(text_span: Span, span: Span)
      #_validate_spans(self, spans: Tuple[Span])
      #_validate_facts(self)
      +get_word(self, span: Span)
      +add_relation_facts(self, facts: Iterable[RelationFact])
   }
   
   class AbstractDataset{
      <<Abstract>>
      +evaluation: bool
      +extract_labels: bool
      +tokenizer
      +max_len: int
      #_documents: List[PreparedDocument]
      #_setup_len_attr(self, tokenizer)
      #_prepare_doc(self, doc: Document)
      +__getitem__(self, idx: int) PreparedDocument
   }
   AbstractDataset ..> Document : processes
   AbstractDataset "1" --o "1..*" PreparedDocument : stores
   
   class PreparedDocument{
      <<NamedTuple>>
      +features: Dict[str, torch.Tensor]
      +labels: Optional[Dict[str, torch.Tensor]]
   }
```

### Models
```mermaid
classDiagram
direction TB

   class TorchModel{
      <<Abstract>>
      +device: torch.device
      +save(self, path: Path, *, rewrite: bool)
      +load(cls, path: Path) TorchModel
   }
   TorchModel <|-- AbstractModel
   AbstractModel <|-- AbstractWrapperModel
   
   class AbstractModel{
      <<Abstract>>
      +relations: Tuple[str]
      +forward(self, *args, **kwargs) Any
      +prepare_dataset(self, documents: Iterable[Document], extract_labels, evaluation)  AbstractDataset
   }
   
   class AbstractWrapperModel{
      <<Abstract>>
      +evaluate(self, dataloader: DataLoader, output_path: str)
      +predict(self, documents: Iterable[Document], dataloader: DataLoader, output_path: str)
      +test(self, dataloader: DataLoader, output_path: str)
   }
   
```

## Run


### Build docker container
1) `cd path/to/project`
2) `docker build ./`
3) `docker run -it --gpus=all __image_id__ /bin/bash`

### Dowload datasets

`bash scripts/download_datasets.sh`

### Start training

`bash scripts/main.sh -c path/to/config -v __gpu_id__`
