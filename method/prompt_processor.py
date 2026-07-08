class PromptProcessor:
    def __init__(self, prompt_type, with_MediaBG = False):
        self.prompt_type = prompt_type
        self.with_MediaBG = with_MediaBG
        self.initial_prompt =  "You are given a question and several pieces of evidence. Your task is to analyze the evidence and provide a concise answer to the question. \n\n"
        if self.with_MediaBG:
            self.initial_prompt += (
                "For each piece of evidence, the background of its source media is provided. "
                "When evaluating the evidence, it is crucial to take into account the credibility of the source media, "
                "as this can significantly influence the reliability of the evidence. Additionally, consider any potential biases "
                "that may be inherent in the source media, especially if they are explicitly mentioned. This will help ensure a "
                "more nuanced and thorough evaluation of the evidence, factoring in both the content and the context in which it is presented.\n\n"
            )

    def generate_prompt(self, question, contexts):
            
        if self.prompt_type == "DirectAnswer":
            return self._generate_direct_answer_prompt(question, contexts)

        elif self.prompt_type == "Explain":
            return self._generate_explain_prompt(question, contexts)

        elif self.prompt_type == "CoT":
            return self._generate_CoT_prompt(question, contexts)
        
        elif self.prompt_type == "ConflictLoc":
            return self._generate_conflictloc_prompt(question, contexts)

        elif self.prompt_type == "ConflictLocExact":
            return self._generate_conflictloc_exact_prompt(question, contexts)
        
        elif self.prompt_type == "ConflictLocClaim":
            return self._generate_conflictloc_claim_prompt(question, contexts)
        
        elif self.prompt_type == "ConflictLocEvidence":
            return self._generate_conflictloc_evidence_prompt(question, contexts)
        
        elif self.prompt_type == "ConflictLocSoft":
            return self._generate_conflictloc_soft_prompt(question, contexts)

        else:
            raise ValueError(f"Invalid prompt type: {self.prompt_type}")
        
    def generate_prompts(self, entities):

        return [self.generate_prompt(entity['question'], entity['top_k']) for entity in entities]

    def _generate_direct_answer_prompt(self, question, contexts):
        prompt = self.initial_prompt

        for idx, evidence in enumerate(contexts):
            prompt += f"- Evidence {idx + 1}:\n{evidence['sentence']}\n"
            if self.with_MediaBG:
                prompt += f"- Source Media Background: {evidence['details']}\n\n"

        prompt += f"\nQuestion: {question}\n\nBased on the evidence, answer the question. Your question should only be 'yes' or 'no'.\nYour answer:"

        return prompt


    def _generate_explain_prompt(self, question, contexts):

        prompt = self.initial_prompt


        prompt += f"\nQuestion: {question}\n\n"


        prompt += "Some evidence below may have been perturbed with wrong information. Find the perturbed passages and ignore them when eliciting the correct answer.\n"

        for idx, context in enumerate(contexts):
            if self.with_MediaBG:
                
                prompt += (
                    f"Evidence {idx + 1}:\n{context['sentence']}\n"
                    f"Source Media Description: {context.get('details', 'None')}\n\n"
                )
            else:
                prompt += (
                    f"Evidence {idx + 1}:\n{context['sentence']}\n"
                )


        prompt += (
            "\n\nFirst, thoroughly analyze all the provided evidence before making your final decision. "
            "Identify the perturbed sentences and carefully consider their implications in your analysis. "
            "Once you have completed your review, provide your final answer to the question based on the evidence you analyzed. "
            "Start your answer with 'Final Answer:' and ensure it is clearly separated from your evidence analysis.\n"
            "Your final answer should be either 'yes' or 'no'.\n"
            "Make sure to include only one final answer, and do not include any additional text after it.\n\n"
        )

        return prompt

    
    def _generate_CoT_prompt(self, question, contexts):

        prompt = self.initial_prompt

        prompt += f"Question: {question}\n\n"
 
        for idx, context in enumerate(contexts):
            if self.with_MediaBG:
                prompt += (
                    f"Evidence {idx + 1}:\n{context['sentence']}\n"
                    f"Source Media Description: {context.get('details', 'None')}\n\n"
                )
            else:

                prompt += (
                    f"Evidence {idx + 1}:\n{context['sentence']}\n\n"
                )

        prompt += (
            "\nGiven the above evidence, first explain your reasoning for any contradictions or conflicting information. "
            "After your reasoning, provide your final answer to the question. "
            "Start your answer with '#*# Final Answer' and clearly separate it from the rest of your analysis.\n"
            "Your final answer should be either 'yes' or 'no'. "
            "Include only one final answer, and avoid adding any additional explanation after it.\n\n"
            "Output format:\n"
            "Analysis: [Your reasoning here] #*# Final Answer: [yes/no]\n\n"
        )

        return prompt
    

    def _generate_conflictloc_prompt(self, question, contexts):
        prompt = self.initial_prompt

        prompt += f"Question: {question}\n\n"

        for idx, context in enumerate(contexts):
            if self.with_MediaBG:
                prompt += (
                    f"Evidence {idx + 1}:\n{context['sentence']}\n"
                    f"Source Media Description: {context.get('details', 'None')}\n\n"
                )
            else:
                prompt += (
                    f"Evidence {idx + 1}:\n{context['sentence']}\n\n"
                )
        
        prompt += (
            "Before answering, briefly check whether the evidence exactly matches the question. "
            "Pay special attention to mismatches in entity, event, cause, time, location, number, or scope. "
            "If different evidence conflicts, prefer the more credible and relevant source based on the media descriptions. "
            "Then answer the question.\n\n"
            "Start your answer with 'Final Answer:' and ensure it is clearly separated from your evidence analysis.\n"
            "Your final answer should be either 'yes' or 'no'.\n"
            "Make sure to include only one final answer, and do not include any additional text after it.\n\n"
        )
        
        return prompt

    def _conflictloc_footer(self):
        # SBAexp 원문 형식 그대로 (파싱 정합 보장)
        return (
            "\n\nStart your answer with 'Final Answer:' and ensure it is clearly separated from your evidence analysis.\n"
            "Your final answer should be either 'yes' or 'no'.\n"
            "Make sure to include only one final answer, and do not include any additional text after it.\n\n"
        )

    def _conflictloc_body(self, question, contexts):
        prompt = f"\nQuestion: {question}\n\n"
        for idx, context in enumerate(contexts):
            if self.with_MediaBG:
                prompt += (
                    f"Evidence {idx + 1}:\n{context['sentence']}\n"
                    f"Source Media Description: {context.get('details', 'None')}\n\n"
                )
            else:
                prompt += f"Evidence {idx + 1}:\n{context['sentence']}\n\n"
        return prompt

    def _generate_conflictloc_exact_prompt(self, question, contexts):
        prompt = self.initial_prompt + self._conflictloc_body(question, contexts)
        prompt += (
            "Before answering, check whether the evidence supports the exact claim rather than only a "
            "topically related claim. Minor paraphrases count as matches, but core mismatches in entity, "
            "event, cause, time, location, number, or scope should affect the verdict."
        )
        prompt += self._conflictloc_footer()
        return prompt

    def _generate_conflictloc_claim_prompt(self, question, contexts):
        prompt = self.initial_prompt + self._conflictloc_body(question, contexts)
        prompt += (
            "Before answering, for each evidence briefly judge whether it SUPPORTS, REFUTES, or is NEUTRAL "
            "toward the exact claim. Minor paraphrases count as support, but if an evidence mismatches the "
            "claim's core conditions, do not count it as support."
        )
        prompt += self._conflictloc_footer()
        return prompt

    def _generate_conflictloc_evidence_prompt(self, question, contexts):
        prompt = self.initial_prompt + self._conflictloc_body(question, contexts)
        prompt += (
            "Before answering, check whether any pieces of evidence conflict with each other. If they do, "
            "prefer the more credible and relevant source based on the media descriptions when deciding the verdict."
        )
        prompt += self._conflictloc_footer()
        return prompt

    def _generate_conflictloc_soft_prompt(self, question, contexts):
        # 현재 v2를 과교정되지 않게 부드럽게
        prompt = self.initial_prompt + self._conflictloc_body(question, contexts)
        prompt += (
            "Before answering, check whether the evidence supports the exact claim, not only a topically "
            "related one. Minor paraphrases count as matches; only core mismatches in entity, event, cause, "
            "time, location, number, or scope should affect the verdict. If different evidence conflicts, "
            "prefer the more credible and relevant source based on the media descriptions."
        )
        prompt += self._conflictloc_footer()
        return prompt