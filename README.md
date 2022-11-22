# Change of Relation Extraction's Entity Domain

Relation extraction (RE) is the task of discovering entities' relations in weakly structured text. There is a lot of applications for RE 
such as knowledge-base population, question answering, summarization and so on. However, despite the increasing number of studies, 
there is a lack of cross-domain evaluation researches. The purpose of this work is to explore how models can be adapted to the changing 
types of entities.

## Motivation
There are several ways to deal with changing entities' types
1) Ignoring (baseline)
   
    * Build a model that does not use any information of entities' types (get lower results)
    * Or don't pay any attention to the domain shift during inference (get lower results)
    
2) Mapping

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
   subgraph c["Model domain"]
      direction TB
      subgraph d[" "]
         direction LR
         time([TIME])-- Date of birth --->person1(["PERSON"])
         number([NUMBER])-- Age --->person2(["PERSON"])
      end
   end
   a ==> c 
```

4) Adopting
