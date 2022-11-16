var fs = require('fs')
var {cosinesim, needleman_wunsch_align, jensen_shannon} = require('./node_compare')

function get_logit_chars(logits, alphabet) {
    let logit_chars = [];
    for (let i = 0; i < logits.length; i ++) {
        let max_logit = 0.0;
        let max_index = 0;
        for (let j = 0; j < logits[i].length; j++) {
            if (logits[i][j] > max_logit) {
                max_logit = logits[i][j];
                max_index = j;
            }
        }
        logit_chars[i] = alphabet[max_index];
    }
    return logit_chars;
}

function clean_text(text) {
    return text.toLowerCase().replace(/\.|,|\?|!|—|’|;|-|:|"|\//gi, '');
}

// returns true if the file exists, false otherwise
// async function check_file_exists(filename) {
//     let http = new XMLHttpRequest();
//     http.open('HEAD', filename, false);
//     http.send();
//     return http.status!==404;
// }

async function run_compare() {
    // initialize variables
    let filename = "";
    let gold_filename = "";
    // let feedback_table = document.getElementById("feedback_table");
    // let colgroup = document.getElementById("table_colgroup");
    // let text_row = document.getElementById("text");
    // let gold_row = document.getElementById("gold_scores");
    // let test_row = document.getElementById("test_scores");
    // let cosine_row = document.getElementById("cosine");
    // let js_row = document.getElementById("js_row");
    let alphabet_index = null;
    let skipped_files = 0;

    // reset table
    // colgroup.innerHTML = "<col>";
    // text_row.innerHTML = "<td>Sentence: </td>";
    // gold_row.innerHTML = "<td>Gold Scores: </td>";
    // test_row.innerHTML = "<td>Test Scores: </td>";
    // cosine_row.innerHTML = "<td>Cosine Scores:</td>";
    // js_row.innerHTML = "<td>JSD Scores:</td>";

    console.log("reading tsv")
    let tsv = [];
    let value = fs.readFileSync("../STT/data/en/sampled_duplicates.tsv");
    let rows = ('' + value).trim().split('\n');
    for(let i = 0; i < rows.length; i++) {
        // client_id	path	sentence	up_votes	down_votes	age	gender	accents	locale	segment
        tsv[i] = rows[i].split('\t');
    }

    console.log("running comparisons");
    for(let row_i = 0; row_i < (tsv.length -1); row_i ++) {
        //console.log(row_i)
        //
        // if (!await check_file_exists("../STT/data/en/clips/" + tsv[row_i][1].slice(0, -3) + "wav")) {
        //     continue;
        // }
        try {
            let stats = fs.statSync("../STT/data/en/clips/" + tsv[row_i][1].slice(0, -3) + "wav");
            // console.log(stats)
        } catch (err) {
            skipped_files++;
            //console.log("skipping file " + "../STT/data/en/clips/" + tsv[row_i][1].slice(0, -3) + "wav")
            continue;
        }
        let gold_phrase = clean_text(tsv[row_i][2]);
        gold_filename = "../STT/data/en/json/" + tsv[row_i][1] + ".json";
        const gold = JSON.parse(fs.readFileSync(gold_filename, 'utf8').replace(/\t/g, "\\t"));
        
        //console.log("Gold:")
        //console.log(gold)

        let row_j = row_i + 1;
        // console.log(row_j + ": " + clean_text(tsv[row_j][2]));
        while (row_j < tsv.length && clean_text(tsv[row_j][2]) === gold_phrase) {
            //console.log(row_j);
            // if (!await check_file_exists("../STT/data/en/clips/" + tsv[row_j][1].slice(0, -3) + "wav")) {
            //     continue;
            // }
            try {
                fs.statSync("../STT/data/en/clips/" + tsv[row_j][1].slice(0, -3) + "wav");
            } catch (err) {
                row_j++;
                continue;
            }
            filename = "../STT/data/en/json/" + tsv[row_j][1] + ".json";
            console.log("comparing files: " + gold_filename + " and " + filename);
            // load data from json files
            const test = JSON.parse(fs.readFileSync(filename, 'utf8').replace(/\t/g, "\\t"));
            
            // const test = await fetch(filename)
            //     .then(function (response) {
            //         return response.text().then(function (value) {
            //             return JSON.parse(value.replace(/\t/g, "\\t"));
            //         })
            //     });
            // const gold = await fetch(gold_filename)
            //     .then(function (response) {
            //         return response.text().then(function (value) {
            //             return JSON.parse(value.replace(/\t/g, "\\t"));
            //         })
            //     });

            // retrieve gold text from gold json
            // let gold_phrase = "";
            // for (let i = 0; i < gold.words.length; i++) {
            //     gold_phrase += gold.words[i].word + (i === gold.words.length - 1 ? '' : ' ');
            // }

            // map alphabet characters to their index (only once)
            if (alphabet_index === null) {
                alphabet_index = new Map();
                for (let i = 0; i < gold.alphabet.length; i++) {
                    alphabet_index.set(gold.alphabet[i], i);
                }
            }

            // set cells to gold text
            // for (let i = 0; i < gold_phrase.length; i++) {
            //     let new_cell = text_row.insertCell();
            //     new_cell.innerText = gold_phrase[i];
            // }

            // get logit characters
            console.log("converting to characters");
            let gold_logit_chars = get_logit_chars(gold.emissions, gold.alphabet);
            let test_logit_chars = get_logit_chars(test.emissions, test.alphabet);

            // retrieve alignments
            console.log("aligning");
            let gold_align = needleman_wunsch_align(gold_phrase, gold_logit_chars);
            let test_align = needleman_wunsch_align(gold_phrase, test_logit_chars);
            // let cosine_align = needleman_wunsch_align(gold.emissions, test.emissions, cosinesim);
            // console.log(gold_align);
            // console.log(test_align);
            // console.log(cosine_align);

            let best_golds = [];
            let best_tests = [];
            let best_gold_indices = [];

            console.log("scoring")
            // get highest score for each character in gold
            let c = 0;
            let c_index = alphabet_index.get(gold_phrase[0]);
            // have to create array of length for comparitor using a[c_index]
            let a = new Array(gold.emissions[0].length).fill(0.0);
            let a_index = 0;
            for (let i = 0; i < gold_align.length; i++) {
                if (gold_align[i][0] !== c) { // save best gold and increment c
                    best_golds[c] = a;
                    best_gold_indices[c] = a_index;
                    a = new Array(gold.emissions[0].length).fill(0.0);
                    c++;
                    c_index = alphabet_index.get(gold_phrase[c]);
                }
                // console.log(c + ': ' + c_index);
                // console.log(gold.emissions[gold_align[i][1]]);
                if (gold.emissions[gold_align[i][1]][c_index] > a[c_index]) {
                    a = gold.emissions[gold_align[i][1]];
                    a_index = i;
                }
                // if (Math.max(...gold.emissions[gold_align[i][1]]) > Math.max(...a)) {
                //     a = gold.emissions[gold_align[i][1]];
                // }
            }
            best_golds[c] = a;
            best_gold_indices[c] = a_index;

            // get highest score for each character in test
            c = 0;
            c_index = alphabet_index.get(gold_phrase[0]);
            a = new Array(gold.emissions[0].length).fill(0.0);
            for (let i = 0; i < test_align.length; i++) {
                if (test_align[i][0] !== c) { // save best test and increment c
                    best_tests[c] = a;
                    a = new Array(gold.emissions[0].length).fill(0.0);
                    c++;
                    c_index = alphabet_index.get(gold_phrase[c]);
                }
                // console.log(c + ': ' + c_index);
                // console.log(test.emissions[test_align[i][1]]);
                if (test.emissions[test_align[i][1]][c_index] > a[c_index]) {
                    a = test.emissions[test_align[i][1]];
                }
                // if (Math.max(...test.emissions[test_align[i][1]]) > Math.max(...a)) {
                //     a = test.emissions[test_align[i][1]];
                // }
            }
            best_tests[c] = a;

            console.log("getting scores")
            let best_golds_print = []
            // insert the gold characters into the table
            for (let i = 0; i < best_golds.length; i++) {
                // let new_cell = gold_row.insertCell();
                // let new_col = document.createElement("col");
                // new_col.style.backgroundColor = "rgba(255, 0, 0, " + (1 - best_golds[i][alphabet_index.get(gold_phrase[i])]) + ")"
                // colgroup.append(new_col);
                // new_cell.innerText = cosinesim(best_golds[i], best_tests[i]).toFixed(3);
                best_golds_print[i] = best_golds[i][alphabet_index.get(gold_phrase[i])].toFixed(3);
            }

            let best_tests_print = []
            // insert the raw max test scores into the table
            for (let i = 0; i < best_tests.length; i++) {
                // let new_cell = test_row.insertCell();
                // let new_col = document.createElement("col");
                // if (gold_phrase[i] !== ' ') {
                //     new_col.style.backgroundColor = "rgba(255, 0, 0, " + Math.max(.6 - best_tests[i][alphabet_index.get(gold_phrase[i])], 0) + ")"
                // }
                // colgroup.append(new_col);
                best_tests_print[i] = best_tests[i][alphabet_index.get(gold_phrase[i])].toFixed(3);
                // new_cell.style.backgroundColor = "rgba(255, 0, 0, " + Math.max(.6 - best_tests[i][alphabet_index.get(gold_phrase[i])], 0) + ")"
            }

            let best_cos_print = []
            // naively insert cosine sim
            for (let i = 0; i < best_golds.length; i++) {
                // let new_cell = cosine_row.insertCell();
                let cos = cosinesim(best_golds[i], best_tests[i]);
                // let best_fixed = [];
                // let test_fixed = [];
                // for (let j = 0; j < best_golds[i].length; j++) {
                //     best_fixed.push(best_golds[i][j].toFixed(20));
                //     test_fixed.push(best_tests[i][j].toFixed(20));
                // }
                // console.log('Gold at ' + i + ': ' + best_fixed);
                // // console.log(best_golds[i]);
                // console.log('Test at ' + i + ': ' + test_fixed);
                // // console.log(best_tests[i]);
                // console.log('cos at ' + i + ': ' + cos);
                best_cos_print[i] = cos.toFixed(3);
                // new_cell.style.backgroundColor = "rgba(255, 0, 0, " + Math.max((.6 - cos), 0) + ")"
            }

            let best_jsd_print = []
            // naively insert Jensen-Shannon
            for (let i = 0; i < best_golds.length; i++) {
                // let new_cell = js_row.insertCell();
                let jsd = jensen_shannon(best_golds[i], best_tests[i]);
                best_jsd_print[i] = (1 - jsd).toFixed(3);
                // new_cell.style.backgroundColor = "rgba(255, 0, 0, " + Math.max((.6 - (1 - jsd)), 0) + ")"
            }

            // print results to file
            let print_text = best_golds_print.join(',') + '\n' + best_tests_print.join(',') + '\n' +
                best_cos_print.join(',') + '\n' + best_jsd_print.join(',');
            let new_file = tsv[row_i][1] + '.' + tsv[row_j][1];
            console.log("writing to " + new_file);
            fs.writeFileSync('../STT/data/en/results/' + new_file + '.txt', print_text);

            //         for (let i = 0; i < best_golds.length; i++) {
            //             let new_cell = feedback_row.insertCell(i);
            //             // new_cell.innerText = cosinesim(best_golds[i], best_tests[i]);
            //             let colour = 'rgb(0,' + (Math.max(...best_golds[i]) * 200) + ',0)';
            //                let text = Math.max(...best_golds[i]).toFixed(2);
            //                new_cell.innerHTML = '<span style="color:' + colour + '">' + text + '</span>';
            //                new_cell.innerText = Math.max(...best_golds[i]).toFixed(2);
            //         }
            //console.log("incrementing row_j");
            row_j++;
        }
    }
    console.log("Finished")
    console.log("Files Skipped: " + skipped_files);
}

run_compare()
