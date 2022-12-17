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

### Features
```mermaid
classDiagram
direction TB

   AbstractFact <|-- EntityFact
   AbstractFact <|-- RelationFact
   EntityFact "1" --> "1..*" Span : is mentioned in
   AbstractFact "1" --> "1" FactType : is a
   
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
      +coreference_id: str
      +mentions: Tuple[Span]
      -validate_mentions(self)
   }
   
   class RelationFact{
      +from_fact: EntityFact
      +to_fact: EntityFact
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
      +max_len
      #_setup_len_attr(self, tokenizer)
      #_prepare_doc(self, doc: Document)
   }
```

### Models
```mermaid
classDiagram
direction TB

   class TorchModel{
      <<Abstract>>
      +device: torch.device
      +forward(self, *args, **kwargs) Any
      +to_device(self, tensor: Tensor) Tensor
      +save(self, path: Path, *, rewrite: bool)
      +load(cls, path: Path) TorchModel
      +from_config(cls, config: dict) TorchModel
   }
   TorchModel <|-- AbstractModel
   
   class AbstractModel{
      <<Abstract>>
      +entities: Tuple[str]
      +relations: Tuple[str]
      +no_ent_ind: int
      +no_rel_ind: int
      +prepare_dataset(self, documents: Iterable[Document], extract_labels, evaluation)  AbstractDataset
   }
   
```
