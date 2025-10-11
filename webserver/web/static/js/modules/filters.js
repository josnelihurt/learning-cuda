/**
 * FilterManager - Manages filter selection and configuration
 */
class FilterManager {
    constructor() {
        this.filterOrder = ['grayscale'];
        this.initDragAndDrop();
    }
    
    getSelectedFilters() {
        const filters = [];
        const grayscaleCheckbox = document.getElementById('filterGrayscale');
        
        if (grayscaleCheckbox && grayscaleCheckbox.checked) {
            filters.push('grayscale');
        }
        
        return filters.length > 0 ? filters : ['none'];
    }
    
    getGrayscaleType() {
        const selected = document.querySelector('input[name="grayscale-algo"]:checked');
        return selected ? selected.value : 'bt601';
    }
    
    toggleCard(header) {
        const card = header.parentElement;
        const body = card.querySelector('.filter-body');
        
        if (card.classList.contains('disabled')) return;
        
        const isExpanded = body.classList.contains('expanded');
        
        if (isExpanded) {
            body.classList.remove('expanded');
            header.classList.add('collapsed');
        } else {
            body.classList.add('expanded');
            header.classList.remove('collapsed');
        }
    }
    
    updateFiltersUI() {
        // Auto-expand/collapse based on checkbox
        const grayscaleCheckbox = document.getElementById('filterGrayscale');
        if (grayscaleCheckbox) {
            const card = grayscaleCheckbox.closest('.filter-card');
            const body = card.querySelector('.filter-body');
            const header = card.querySelector('.filter-header');
            
            if (grayscaleCheckbox.checked) {
                body.classList.add('expanded');
                header.classList.remove('collapsed');
            } else {
                body.classList.remove('expanded');
                header.classList.add('collapsed');
            }
        }
    }
    
    /* ============================================
       DRAG & DROP
       ============================================ */
    
    initDragAndDrop() {
        this.draggedElement = null;
        this.draggedIndex = null;
        
        document.addEventListener('DOMContentLoaded', () => {
            const filtersList = document.getElementById('filtersList');
            if (!filtersList) return;
            
            const cards = filtersList.querySelectorAll('.filter-card:not(.disabled)');
            
            cards.forEach((card) => {
                card.addEventListener('dragstart', (e) => this.handleDragStart(e, card));
                card.addEventListener('dragend', (e) => this.handleDragEnd(e, card));
                card.addEventListener('dragover', (e) => this.handleDragOver(e));
                card.addEventListener('drop', (e) => this.handleDrop(e, card));
                card.addEventListener('dragenter', (e) => this.handleDragEnter(e, card));
                card.addEventListener('dragleave', (e) => this.handleDragLeave(e, card));
            });
        });
    }
    
    handleDragStart(e, card) {
        this.draggedElement = card;
        this.draggedIndex = Array.from(card.parentNode.children).indexOf(card);
        card.classList.add('dragging');
        e.dataTransfer.effectAllowed = 'move';
    }
    
    handleDragEnd(e, card) {
        card.classList.remove('dragging');
        document.querySelectorAll('.filter-card').forEach(c => c.classList.remove('drag-over'));
    }
    
    handleDragOver(e) {
        e.preventDefault();
        e.dataTransfer.dropEffect = 'move';
        return false;
    }
    
    handleDragEnter(e, card) {
        if (card !== this.draggedElement && !card.classList.contains('disabled')) {
            card.classList.add('drag-over');
        }
    }
    
    handleDragLeave(e, card) {
        card.classList.remove('drag-over');
    }
    
    handleDrop(e, card) {
        e.stopPropagation();
        
        if (this.draggedElement !== card && !card.classList.contains('disabled')) {
            const dropIndex = Array.from(card.parentNode.children).indexOf(card);
            
            if (this.draggedIndex < dropIndex) {
                card.parentNode.insertBefore(this.draggedElement, card.nextSibling);
            } else {
                card.parentNode.insertBefore(this.draggedElement, card);
            }
            
            this.updateFilterOrder();
            console.log('Filter order:', this.filterOrder);
        }
        
        return false;
    }
    
    updateFilterOrder() {
        const filtersList = document.getElementById('filtersList');
        const cards = filtersList.querySelectorAll('.filter-card:not(.disabled)');
        this.filterOrder = Array.from(cards).map(card => card.dataset.filter);
    }
}

