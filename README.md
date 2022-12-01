# Change of Relation Extraction's Entity Domain

Relation extraction (RE) is the task of discovering entities' relations in weakly structured text. There is a lot of
applications for RE such as knowledge-base population, question answering, summarization and so on. However, despite the
increasing number of studies, there is a lack of cross-domain evaluation researches. The purpose of this work is to
explore how models can be adapted to the changing types of entities.

## Motivation

There are several ways to deal with the changing types of entyties:

1) Ignoring (our baseline)

    * Build a model that does not use any information of entities' types (and get lower results);
    * Or don't pay any attention to the domain shift during inference (and also get lower results).

2) Mapping

   Another way is to build a mapping from the model's entity types to another domain ones. But there may be situations
   when it is impossible to build the unambiguous mapping (e.g. diagram below, where `PER` correspond only to `PERSON`,
   but `NUM` is `NUMBER` and `TIME` concurrently).

   In the case of the unambiguous mapping, we can try all suitable mappings, but if there are $N$ entities and $M$
   candidates for each of them, $M^N$ model runs are required.

```mermaid
flowchart LR
    subgraph a["Unknown domain"]
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

3) Adapting

   We are going to develop training methods that instill domain shift resistance in RE models and allow them to adapt to
   new types of entities.

## Project structure

## Class diagrams

The base classes are divided into 3 main categories:

* **_Features_**:
  * Span
  * FactType
  * AbstractFact
    * EntityFact
    * RelationFact
  * CoreferenceChain
* **_Examples_**:
  * Document
  * AbstractDataset
* **_Models_**:
  * TorchModel
  * AbstractModel
  * RelextModel
  * AbstractSubModel

### Features
```mermaid
classDiagram
direction TB

   AbstractFact <|-- EntityFact
   AbstractFact <|-- RelationFact
   EntityFact "1" --> "many" Span : is mentioned in
   AbstractFact "1" --> "1" FactType : is a
   EntityFact "many" --o "1" CoreferenceChain : consists of 
   
   class Span:::rect{
      +start_idx: int
      +end_idx: int
   }
   
   class FactType{
      <<Enumeration>>
      ENTITY
      RELATION
   }

   class AbstractFact{
      <<Abstract>> 
      +fact_id: str
      +fact_type_id: FactType
      +fact_type: str
   }
   
   class EntityFact{
      +mentions: Tuple[Span]
      -validate_mentions(self)
   }
   
   class RelationFact{
      +from_fact: EntityFact
      +to_fact: EntityFact
   }
   
   class CoreferenceChain{
      +facts: Tuple[EntityFact]
   }
```
### Examples
```mermaid
classDiagram
direction TB
   class Document{
      +doc_id: str
      +text: str
      +words: Tuple[Span]
      +sentences: Tuple[Tuple[Span]]
      +facts: Tuple[AbstractFact]
      +coref_chains: Tuple[CoreferenceChain]
      #_validate_span(text_span: Span, span: Span)
      #_validate_spans(spans: Tuple[Span])
      #_validate_chains()
      #_validate_facts()
      +get_word(span: Span)
      +add_relation_facts(facts: Iterable[RelationFact])
   }
   
   class AbstractDataset{
      <<Abstract>>
      +tokenizer
      +evaluation: bool
      +extract_labels: bool
      +get_fact(doc_idx, fact_idx) AbstractFact
      +fact_count(doc_idx) int
      #_prepare_doc(doc: Document)
   }
```

### Models
```mermaid
classDiagram
direction TB

   class TorchModel{
      +device: torch.device
      +forward(*args, **kwargs) Any
      +save(path: Path, *, rewrite: bool)
      +load(path: Path) TorchModel
      +from_config(config: dict) TorchModel
   }
   TorchModel <-- AbstractModel
   
   class AbstractModel{
      <<Abstract>>
      +prepare_dataset(documents: Iterable[Document])  AbstractDataset
   }
   AbstractModel <-- RelextModel
   AbstractModel <-- AbstractSubModel
   
   class RelextModel{
      #_inner_model: AbstractSubModel
      +predict(doc: Document) Document
   }
   RelextModel <-- SSANAdapt
   RelextModel <-- REBEL
   RelextModel "1" --> "0..1" AbstractSubModel: contains
   
   class AbstractSubModel{
      <<Abstract>>
   }
   
```
