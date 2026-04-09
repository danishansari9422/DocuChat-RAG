"""
RAG Evaluation Module

Handles evaluation of RAG system performance using predefined questions.
"""

import pandas as pd
from typing import List, Dict, Any, Optional
from rag.qa_chain import QAChain


class RAGEvaluator:
    """Handles evaluation of RAG system performance."""
    
    def __init__(self, qa_chain: QAChain):
        """
        Initialize the evaluator.
        
        Args:
            qa_chain: QAChain instance for answering questions
        """
        self.qa_chain = qa_chain
        self.test_questions = self._get_test_questions()
        self.expected_answers = self._get_expected_answers()
        self.evaluation_results = []
    
    def _get_test_questions(self) -> List[str]:
        """
        Get predefined test questions.
        
        Returns:
            List of test questions
        """
        return [
            "What is this paper about?",
            "What problem does this paper aim to solve?",
            "What is the win rate of vector-based RAG?",
            "What is cross-encoder reranking?",
            "How does small-to-big retrieval improve performance?"
        ]
    
    def _get_expected_answers(self) -> Dict[str, str]:
        """
        Get expected answers for reference.
        
        Returns:
            Dictionary mapping questions to expected answers
        """
        return {
            "What is this paper about?": "RAG systems in the financial domain, comparing traditional retrieval with modern approaches",
            "What problem does this paper aim to solve?": "Improving RAG performance for financial document analysis",
            "What is the win rate of vector-based RAG?": "Vector-based RAG shows high win rates compared to traditional methods",
            "What is cross-encoder reranking?": "A technique to improve retrieval accuracy by reranking results",
            "How does small-to-big retrieval improve performance?": "Better relevance and efficiency in retrieval by using smaller chunks first"
        }
    
    def run_evaluation(self, progress_callback=None) -> List[Dict[str, Any]]:
        """
        Run complete evaluation on all test questions.
        
        Args:
            progress_callback: Optional callback for progress updates
            
        Returns:
            List of evaluation results
        """
        results = []
        total_questions = len(self.test_questions)
        
        for i, question in enumerate(self.test_questions):
            if progress_callback:
                progress_callback(i, total_questions, question)
            
            try:
                # Get answer from RAG system
                rag_result = self.qa_chain.answer_question(question)
                model_answer = rag_result['answer']
                citations = rag_result.get('citations', [])
                
                # Get expected answer
                expected_answer = self.expected_answers.get(question, "No expected answer")
                
                result = {
                    'question': question,
                    'model_answer': model_answer,
                    'expected_answer': expected_answer,
                    'citations': citations,
                    'correct': None,  # To be set manually
                    'notes': '',
                    'context_used': rag_result.get('context_used', False)
                }
                
                results.append(result)
                
            except Exception as e:
                # Handle errors gracefully
                result = {
                    'question': question,
                    'model_answer': f"Error: {str(e)}",
                    'expected_answer': self.expected_answers.get(question, "No expected answer"),
                    'citations': [],
                    'correct': False,
                    'notes': f"System error: {str(e)}",
                    'context_used': False
                }
                
                results.append(result)
        
        self.evaluation_results = results
        return results
    
    def update_result(self, question_index: int, correct: bool, notes: str = ""):
        """
        Update evaluation result for manual marking.
        
        Args:
            question_index: Index of the question to update
            correct: Whether the answer is correct
            notes: Additional notes
        """
        if 0 <= question_index < len(self.evaluation_results):
            self.evaluation_results[question_index]['correct'] = correct
            self.evaluation_results[question_index]['notes'] = notes
    
    def calculate_accuracy(self) -> Dict[str, Any]:
        """
        Calculate evaluation metrics.
        
        Returns:
            Dictionary with accuracy metrics
        """
        if not self.evaluation_results:
            return {
                'accuracy': 0.0,
                'total_correct': 0,
                'total_questions': 0,
                'correct_rate': '0.0%'
            }
        
        total_questions = len(self.evaluation_results)
        marked_correct = sum(1 for result in self.evaluation_results if result.get('correct') is True)
        marked_incorrect = sum(1 for result in self.evaluation_results if result.get('correct') is False)
        unmarked = total_questions - marked_correct - marked_incorrect
        
        accuracy = (marked_correct / total_questions * 100) if total_questions > 0 else 0.0
        
        return {
            'accuracy': accuracy,
            'total_correct': marked_correct,
            'total_incorrect': marked_incorrect,
            'unmarked': unmarked,
            'total_questions': total_questions,
            'correct_rate': f"{accuracy:.1f}%"
        }
    
    def get_results_dataframe(self) -> pd.DataFrame:
        """
        Get evaluation results as pandas DataFrame.
        
        Returns:
            DataFrame with evaluation results
        """
        if not self.evaluation_results:
            return pd.DataFrame()
        
        # Prepare data for DataFrame
        df_data = []
        for i, result in enumerate(self.evaluation_results):
            df_data.append({
                'ID': i + 1,
                'Question': result['question'],
                'Model Answer': result['model_answer'][:200] + "..." if len(result['model_answer']) > 200 else result['model_answer'],
                'Expected Answer': result['expected_answer'],
                'Correct': result['correct'],
                'Notes': result['notes'],
                'Context Used': result['context_used']
            })
        
        return pd.DataFrame(df_data)
    
    def export_results(self, filepath: str) -> bool:
        """
        Export evaluation results to CSV.
        
        Args:
            filepath: Path to save the CSV file
            
        Returns:
            True if export successful, False otherwise
        """
        try:
            df = self.get_results_dataframe()
            df.to_csv(filepath, index=False)
            return True
        except Exception as e:
            print(f"Error exporting results: {e}")
            return False
    
    def get_summary_stats(self) -> Dict[str, Any]:
        """
        Get summary statistics of evaluation.
        
        Returns:
            Dictionary with summary statistics
        """
        metrics = self.calculate_accuracy()
        
        # Add additional stats
        context_used_count = sum(1 for result in self.evaluation_results if result.get('context_used', False))
        avg_answer_length = 0
        
        if self.evaluation_results:
            answer_lengths = [len(result['model_answer']) for result in self.evaluation_results]
            avg_answer_length = sum(answer_lengths) / len(answer_lengths)
        
        return {
            **metrics,
            'context_used_count': context_used_count,
            'context_usage_rate': f"{(context_used_count / len(self.evaluation_results) * 100):.1f}%" if self.evaluation_results else "0.0%",
            'avg_answer_length': f"{avg_answer_length:.0f} characters"
        }
