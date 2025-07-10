# Name Processing Challenges and Errors

## Challenge 1: Hindi Character Misrecognition

```
श्री संतोष कुमार WA,Santosh Kumar Kushwaha,संतोष कुमार कुशवाहा

श्री संतोष कुमार गंगवार,Santosh Kumar Gangwar,संतोष कुमार गंगवार
```


Hindi character misrecognition might lead to mischaracterization of names, especially when multiple individuals have very similar names. OCR inaccuracies in Devanagari can cause transliteration mismatches and identity confusion.


---

## Challenge 2: Tabulation Cross-Column Contamination

```
Shri P.K. Biju, श्री रवनीत सिंह (लुधियाना)",Ravneet Singh Bittu
```


In some cases, the tabulation process incorrectly merges adjacent column entries, treating names from a different column as part of the speaker’s name. While no specific error occurred in this example, the structure makes such mistakes possible and difficult to detect automatically.


---

## Challenge 3: Multiple Names for the Same Individual

```
Afrin Ali née Aparupa Poddar
```


Some politicians are known by multiple names (e.g., maiden vs married name), and these inconsistencies between datasets and official documents hinder reliable name matching.




## Key Error Sources


- OCR Limitations: Character recognition failures in Devanagari/Hindi scripts lead to fragmented, incomplete, or wrong name extraction.
- Lack of Standardization: Variations in naming conventions across different documents result in inconsistent representations of the same individual.
- Formatting Differences: Differences in how names are ordered or formatted in official contribute to difficulty in matching.

