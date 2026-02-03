"""Export captions to various formats."""

import csv
import json
from pathlib import Path
from datetime import datetime
from typing import List, Dict, Any, Optional
import logging

logger = logging.getLogger(__name__)


class CaptionExporter:
    """Handle exporting captions to various formats."""
    
    def __init__(self, output_directory: Optional[Path] = None):
        """
        Initialize exporter.
        
        Args:
            output_directory: Directory for output files. If None, uses input directory.
        """
        self.output_directory = output_directory
    
    def export_individual_txt(self, results: List[Dict[str, Any]], output_dir: Optional[Path] = None) -> int:
        """
        Export captions as individual .txt files (one per image).
        
        Args:
            results: List of result dictionaries with 'path' and 'caption'
            output_dir: Output directory (overrides constructor setting)
            
        Returns:
            Number of files written
        """
        output_dir = output_dir or self.output_directory
        count = 0
        
        for result in results:
            image_path = result.get('path')
            caption = result.get('caption', '')
            
            if not image_path or not caption:
                continue
            
            try:
                # Determine output path
                if output_dir:
                    txt_path = output_dir / f"{image_path.stem}.txt"
                else:
                    # Save next to image
                    txt_path = image_path.with_suffix('.txt')
                
                # Write caption
                txt_path.write_text(caption, encoding='utf-8')
                count += 1
                logger.debug(f"Wrote caption to {txt_path}")
                
            except Exception as e:
                logger.error(f"Failed to write {image_path.name}.txt: {e}")
        
        logger.info(f"Exported {count} individual .txt files")
        return count
    
    def export_csv(self, results: List[Dict[str, Any]], output_path: Path, relative_to: Optional[Path] = None) -> bool:
        """
        Export captions as CSV file.
        
        Args:
            results: List of result dictionaries
            output_path: Path to output CSV file
            relative_to: Base path for relative paths (if None, uses absolute paths)
            
        Returns:
            True if successful, False otherwise
        """
        try:
            # Ensure output directory exists
            output_path.parent.mkdir(parents=True, exist_ok=True)
            
            with open(output_path, 'w', newline='', encoding='utf-8') as f:
                fieldnames = [
                    'filepath',
                    'filename',
                    'caption',
                    'timestamp',
                    'success'
                ]
                
                writer = csv.DictWriter(f, fieldnames=fieldnames)
                writer.writeheader()
                
                for result in results:
                    image_path = result.get('path')
                    caption = result.get('caption', '')
                    success = result.get('success', True)
                    
                    if not image_path:
                        continue
                    
                    # Format path
                    filepath = str(image_path)
                    if relative_to:
                        try:
                            # is_relative_to available in Python 3.9+
                            if image_path.is_relative_to(relative_to):
                                filepath = str(image_path.relative_to(relative_to))
                        except (ValueError, AttributeError):
                            # Fallback for Python < 3.9
                            try:
                                filepath = str(image_path.relative_to(relative_to))
                            except ValueError:
                                filepath = str(image_path)
                    
                    writer.writerow({
                        'filepath': filepath,
                        'filename': image_path.name,
                        'caption': caption,
                        'timestamp': datetime.now().isoformat(),
                        'success': success
                    })
            
            logger.info(f"Exported CSV to {output_path}")
            return True
            
        except Exception as e:
            logger.error(f"Failed to export CSV: {e}")
            return False
    
    def export_json(self, results: List[Dict[str, Any]], output_path: Path, metadata: Optional[Dict[str, Any]] = None) -> bool:
        """
        Export captions as JSON file.
        
        Args:
            results: List of result dictionaries
            output_path: Path to output JSON file
            metadata: Optional metadata to include
            
        Returns:
            True if successful, False otherwise
        """
        try:
            # Ensure output directory exists
            output_path.parent.mkdir(parents=True, exist_ok=True)
            
            # Build JSON structure
            output_data = {
                'metadata': metadata or {},
                'generated_at': datetime.now().isoformat(),
                'total_images': len(results),
                'results': []
            }
            
            # Add default metadata
            if 'version' not in output_data['metadata']:
                output_data['metadata']['version'] = '1.0'
            
            # Convert results
            for result in results:
                image_path = result.get('path')
                caption = result.get('caption', '')
                success = result.get('success', True)
                
                if not image_path:
                    continue
                
                output_data['results'].append({
                    'filepath': str(image_path),
                    'filename': image_path.name,
                    'caption': caption,
                    'success': success
                })
            
            # Write JSON
            with open(output_path, 'w', encoding='utf-8') as f:
                json.dump(output_data, f, indent=2, ensure_ascii=False)
            
            logger.info(f"Exported JSON to {output_path}")
            return True
            
        except Exception as e:
            logger.error(f"Failed to export JSON: {e}")
            return False
    
    def export_txt_batch(self, results: List[Dict[str, Any]], output_path: Path) -> bool:
        """
        Export all captions to a single text file.
        
        Args:
            results: List of result dictionaries
            output_path: Path to output text file
            
        Returns:
            True if successful, False otherwise
        """
        try:
            # Ensure output directory exists
            output_path.parent.mkdir(parents=True, exist_ok=True)
            
            with open(output_path, 'w', encoding='utf-8') as f:
                for idx, result in enumerate(results):
                    image_path = result.get('path')
                    caption = result.get('caption', '')
                    
                    if not image_path or not caption:
                        continue
                    
                    # Format: filename on separate line, caption below, blank line separator
                    f.write(f"{image_path.name}\n")
                    f.write(f"{caption}\n")
                    
                    # Add blank line between entries (except after last one)
                    if idx < len(results) - 1:
                        f.write("\n")
            
            logger.info(f"Exported batch text file to {output_path}")
            return True
            
        except Exception as e:
            logger.error(f"Failed to export batch text file: {e}")
            return False
    
    def export_all(self, results: List[Dict[str, Any]], formats: List[str], base_output_path: Path, metadata: Optional[Dict[str, Any]] = None) -> Dict[str, bool]:
        """
        Export captions in multiple formats.
        
        Args:
            results: List of result dictionaries
            formats: List of format names ('txt_individual', 'csv', 'json', 'txt_batch')
            base_output_path: Base path for output files
            metadata: Optional metadata for JSON export
            
        Returns:
            Dictionary mapping format names to success status
        """
        export_status = {}
        
        for fmt in formats:
            try:
                if fmt == 'txt_individual':
                    # Export to same directory as base_output_path
                    count = self.export_individual_txt(results, base_output_path.parent)
                    export_status[fmt] = count > 0
                    
                elif fmt == 'csv':
                    csv_path = base_output_path.parent / f"{base_output_path.stem}_captions.csv"
                    export_status[fmt] = self.export_csv(results, csv_path, relative_to=base_output_path.parent)
                    
                elif fmt == 'json':
                    json_path = base_output_path.parent / f"{base_output_path.stem}_captions.json"
                    export_status[fmt] = self.export_json(results, json_path, metadata)
                    
                elif fmt == 'txt_batch':
                    txt_path = base_output_path.parent / f"{base_output_path.stem}_captions.txt"
                    export_status[fmt] = self.export_txt_batch(results, txt_path)
                    
                else:
                    logger.warning(f"Unknown export format: {fmt}")
                    export_status[fmt] = False
                    
            except Exception as e:
                logger.error(f"Error exporting format {fmt}: {e}")
                export_status[fmt] = False
        
        return export_status
