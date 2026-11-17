"""Finite, objective English tasks, with separate teaching and testing prompts.

The dictionary source is not a philosophical teacher. Exact recall is tested
separately from lexical discrimination, grammar, reading and composition.
No automatic claim is made about unrestricted natural-language composition.
"""
import re
from collections import defaultdict
from hashlib import sha256

STAGES = ("dictionary", "usage", "grammar", "reading", "composition")
CURRICULUM_VERSION = 2
ACTORS = ("child", "farmer", "teacher", "sailor", "clerk", "doctor", "worker", "cook")
OBJECTS = ("book", "key", "cup", "rope", "letter", "bag", "chair", "lamp")
PLACES = ("house", "school", "garden", "shop", "field", "room", "barn", "station")
VERBS = (("carry","carries"),("hold","holds"),("find","finds"),("read","reads"),
         ("open","opens"),("move","moves"),("lift","lifts"),("bring","brings"))

def clean(text):
    return " ".join(text.split())

def words(text):
    return re.findall(r"[^\W\d_]+(?:['’-][^\W\d_]+)*",text.casefold(),re.UNICODE)

def chunks(text, tokenizer, budget):
    """No text discarded: bounded chunks using the actual checkpoint tokenizer."""
    parts=clean(text).split()
    output=[]
    current=""
    for word in parts:
        if len(tokenizer.encode(word))>budget:
            if current:
                output.append(current)
                current=""
            part=""
            for char in word:
                if part and len(tokenizer.encode(part+char))>budget:
                    output.append(part)
                    part=""
                part+=char
            if part:
                output.append(part)
            continue
        candidate=(current+" "+word).strip()
        if current and len(tokenizer.encode(candidate))>budget:
            output.append(current)
            current=word
        else:
            current=candidate
    if current:
        output.append(current)
    if not output:
        raise ValueError("Empty dictionary text")
    return output

class Curriculum:
    def __init__(self, records, tokenizer, context):
        self.records=sorted(records, key=lambda row: row[0].casefold())
        self.tokenizer=tokenizer
        self.context=context
        self.lookup=defaultdict(list)
        for i,(word,_) in enumerate(self.records):
            self.lookup[word.casefold()].append(i)
        self.totals={"grammar":384,"dictionary":len(records),"usage":len(records),
                     "reading":256,"composition":128}
        if not records:
            raise ValueError("Empty dictionary")
    def indices(self, text):
        found=set()
        missing=set()
        for word in words(text):
            variants=[word]
            if word.endswith("'s") or word.endswith("’s"):
                variants.append(word[:-2])
            if word.endswith("s"):
                variants.append(word[:-1])
            if word.endswith("ies"):
                variants.append(word[:-3]+"y")
            if word.endswith("es"):
                variants.append(word[:-2])
            for suffix in ("ed","ing"):
                if word.endswith(suffix):
                    base=word[:-len(suffix)]
                    variants.extend([base,base+"e",base[:-1]])
            ids=next((self.lookup[v] for v in variants if v in self.lookup),[])
            if ids:
                found.update(ids)
            else:
                missing.add(word)
        return sorted(found),sorted(missing)
    def _lex(self, stage, index):
        word, definition=self.records[index]
        label=f"Webster 1913 entry {index+1}: {word}"
        budget=min(80,self.context//4)
        pieces=chunks(definition,self.tokenizer,budget)
        result=[]
        for n,part in enumerate(pieces,1):
            if stage=="dictionary":
                train=f"Write the text of {label}, part {n} of {len(pieces)}."
                exam=f"From memory, reproduce {label}, section {n}/{len(pieces)}."
                answer=part
            else:
                # The selected lexical item is the source entry. Decoys are
                # distinct headwords, not generated alternative definitions.
                others=[]
                for shift in range(1,len(self.records)):
                    candidate=self.records[(index+shift)%len(self.records)][0]
                    if candidate.casefold()!=word.casefold() and candidate not in others:
                        others.append(candidate)
                    if len(others)==2:
                        break
                choices=[word]+others
                rotation=(index+n)%len(choices)
                choices=choices[rotation:]+choices[:rotation]
                choices2=list(reversed(choices))
                train=f"Select the source headword: {part}\nOptions: {' | '.join(choices)}"
                exam=f"Which listed word belongs to this dictionary passage? {part}\nChoices: {' | '.join(choices2)}"
                answer=word
            result.append(dict(id=f"{stage}-{index}-{n}",question=train,exam=exam,answer=answer,lexical_text=word))
        if stage=="dictionary":
            # Complete each headword before moving to the next. Prefer a
            # contextual example in the source; otherwise use an explicitly
            # labelled definitional sentence (not claimed as free usage).
            result.append(self.word_sentence(index, len(pieces)+1))
        return result
    def word_sentence(self, index, part):
        word, definition = self.records[index]
        label = f"Webster 1913 entry {index+1}: {word}"
        pattern = re.compile(r"(?<!\w)"+re.escape(word)+r"(?!\w)", re.I)
        examples = []
        for candidate in re.findall(r'["“]([^"”]+)["”]', clean(definition)):
            candidate = candidate.strip()
            if (pattern.search(candidate) and len(words(candidate))>=4
                    and candidate[0].isupper() and candidate[-1] in ".?!"):
                examples.append(candidate)
        for candidate in examples:
            masked = pattern.sub("___",candidate,count=1)
            question=f"Complete the source sentence using {word}: {masked}"
            exam=f"Restore the missing headword in this source sentence: {masked}"
            if all(len(self.tokenizer.encode(q))+len(self.tokenizer.encode(candidate))+12<=self.context for q in (question,exam)):
                return dict(id=f"word-use-{index}-{part}",question=question,exam=exam,
                    answer=candidate,lexical_text=word,usage_kind="source_example")
        # A dictionary entry may contain only a cross-reference, inflection,
        # abbreviation, or fragment. Preserve the complete original definition
        # in its recall units; use a bounded excerpt for this lexical exercise.
        excerpt=chunks(definition,self.tokenizer,min(40,self.context//8))[0]
        answer=f'The entry "{word}" includes the definition "{excerpt}".'
        question=f"Write a definition sentence for {label}, quoting its opening text."
        exam=f"State the opening dictionary text of {label} in a complete definition sentence."
        if max(len(self.tokenizer.encode(q)) for q in (question,exam))+len(self.tokenizer.encode(answer))+12>self.context:
            raise ValueError("Lexical sentence cannot fit context: "+label)
        return dict(id=f"word-use-{index}-{part}",question=question,exam=exam,
                    answer=answer,lexical_text=word,usage_kind="definition_sentence")
    def units(self, stage, index):
        if not 0<=index<self.totals[stage]:
            raise ValueError("Curriculum cursor escaped")
        if stage in ("dictionary","usage"):
            return self._lex(stage,index)
        a=ACTORS[index%8]
        obj=OBJECTS[(index//8+index)%8]
        place=PLACES[(index//16+index)%8]
        verb,agree=VERBS[(index//8)%8]
        mode=(index//32)%(12 if stage=="grammar" else 8 if stage=="reading" else 4)
        if stage=="grammar":
            if mode==0:
                broken=f"The {a} {verb} the {obj}."
                answer=f"The {a} {agree} the {obj}."
                q=f"Correct the verb agreement: {broken}"
                exam=f"Repair only the subject-verb disagreement in: {broken}"
            elif mode==1:
                broken=f"the {a} {agree} the {obj}"
                answer=broken.capitalize()+"."
                q=f"Add the initial capital and final period: {broken}"
                exam=f"Capitalize and end this statement with a period: {broken}"
            elif mode==2:
                broken=f"The {a}s is in the {place}."
                answer=f"The {a}s are in the {place}."
                q=f"Change only the incorrect verb: {broken}"
                exam=f"Make the verb agree with the plural subject: {broken}"
            elif mode==3:
                broken=f"The {a} did not {agree} the {obj}."
                answer=f"The {a} did not {verb} the {obj}."
                q=f"Repair the verb after did not: {broken}"
                exam=f"Correct only the verb following the auxiliary: {broken}"
            elif mode==4:
                broken=f"The {a} saw a apple near the {place}."
                answer=f"The {a} saw an apple near the {place}."
                q=f"Correct the indefinite article: {broken}"
                exam=f"Replace only the incorrect article in: {broken}"
            elif mode==5:
                broken=f"The {a} saw an {obj}."
                answer=f"The {a} saw a {obj}."
                q=f"Correct the article before the object: {broken}"
                exam=f"Use the right indefinite article without any other change: {broken}"
            elif mode==6:
                broken=f"The {a} and me went to the {place}."
                answer=f"The {a} and I went to the {place}."
                q=f"Correct the subject pronoun: {broken}"
                exam=f"Fix only the pronoun in the compound subject: {broken}"
            elif mode==7:
                broken=f"The {a} gave I the {obj}."
                answer=f"The {a} gave me the {obj}."
                q=f"Correct the object pronoun: {broken}"
                exam=f"Replace the subject pronoun with the appropriate object form: {broken}"
            elif mode==8:
                broken=f"Where is the {obj} in the {place}."
                answer=broken[:-1]+"?"
                q=f"Correct the punctuation at the end of the question: {broken}"
                exam=f"Change only the final punctuation to mark a question: {broken}"
            elif mode==9:
                broken=f"The {a} said I found the {obj}."
                answer=f'The {a} said, "I found the {obj}."'
                q=f'Punctuate the direct quotation starting with I, using double quotation marks: {broken}'
                exam=f'Add a comma and double quotes around the direct speech beginning with I, with the period inside: {broken}'
            elif mode==10:
                broken=f"The {a} will {agree} the {obj}."
                answer=f"The {a} will {verb} the {obj}."
                q=f"Correct the verb after will: {broken}"
                exam=f"Fix only the form of the verb following the modal: {broken}"
            else:
                broken=f"The {a}s {agree} the {obj}."
                answer=f"The {a}s {verb} the {obj}."
                q=f"Correct the plural subject and verb agreement: {broken}"
                exam=f"Repair only the verb for the plural subject: {broken}"
            lexical=broken+" "+answer
        elif stage=="reading":
            passage=f"The {a} left the {obj} in the {place}."
            if mode<4:
                field=("object","place","person","object")[mode]
                answer={"object":obj,"place":place,"person":a}[field]
                q=f"{passage} Name the {field} only."
                exam=f"{passage} Give just the word naming the {field}."
            else:
                positive=index%2==0
                fact=f"The {obj} is "+("ready." if positive else "not ready.")
                passage=f"The {a} may enter the {place} if and only if the {obj} is ready. {fact}"
                answer="yes" if positive else "no"
                q=f"{passage} May the person enter? Reply yes or no."
                exam=f"{passage} Under the rule, is entry permitted? Answer yes or no."
            lexical=passage
        elif stage=="composition":
            # Constrained composition: objectively gradable, not open essays.
            tokens=["the",a,agree,"the",obj]
            answer=" ".join(tokens).capitalize()+"."
            prompt=f"subject=the {a}; verb={verb}; object=the {obj}"
            q=f"Compose a third-person present-tense statement, subject then verb then object: {prompt}"
            exam=f"Write one capitalized present-tense sentence with a full stop, in subject-verb-object order: {prompt}"
            lexical=answer
        else:
            raise ValueError("Unknown English stage")
        unit=dict(id=f"{stage}-{index}",question=q,exam=exam,answer=answer,lexical_text=lexical)
        if stage=="composition":
            # Train multi-sentence construction with one set of facts, test a
            # different combination and wording without exposing the target.
            a2=ACTORS[(index+3)%8]
            o2=OBJECTS[(index//8+index+3)%8]
            p2=PLACES[(index//16+index+5)%8]
            unit["composition_transfer"]=dict(
                exam=f"Write two present-tense sentences. First state that the {a2} possesses the {o2}, using 'has'. Then state where the {o2} is: the {p2}.",
                answer=f"The {a2} has the {o2}. The {o2} is in the {p2}.")
            # Teach the required construction with different nouns and facts.
            unit["composition_practice"]=dict(id=unit["id"]+"-practice",
                question=f"Express possession and location as two sentences (use has and is in): {a}, {obj}, {place}.",
                answer=f"The {a} has the {obj}. The {obj} is in the {place}.")
        return [unit]
    def unit(self, key):
        stage,index,part=key
        return self.units(stage,index)[part]
    def first(self, stage,index):
        return [stage,index,0]
    def next(self, key):
        stage,index,part=key
        if part+1<len(self.units(stage,index)):
            return [stage,index,part+1]
        if index+1<self.totals[stage]:
            return [stage,index+1,0]
        return None
