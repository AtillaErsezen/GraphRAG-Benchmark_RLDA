const { readFileSync, writeFileSync } = require('fs');

for (let i = 1; i < 4; i++) {
    for (let j = 1; j < 4; j++) {
        const file = `results/lightrag_lda_${i}/Medical/light_evaluation_results_generation_medical_${j}.json`;
        const text = readFileSync(file, 'utf8');
        const fixed = text.replace(/\bNaN\b/g, '"NaN"');
        writeFileSync(file, fixed, 'utf8');
        console.log(`Fixed: ${file}`);
    }
}
