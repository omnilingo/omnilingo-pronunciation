var {needleman_wunsch_align, } = require('./compare')

function getRndInteger(min, max) {
    return Math.floor(Math.random() * (max - min) ) + min;
}

function getRndChar() {
    return String.fromCharCode(getRndInteger(97, 122))
}

function makeRandomList(len) {
    let new_list = []
    for (let i = 0; i < len; i++) {
        new_list.push(getRndChar())
    }
    return new_list
}

function run_test() {
    let characters = makeRandomList(getRndInteger(1, 5))
    let candidates = []
    for (let i = 0; i < characters.length; i++) {
        let m = getRndInteger(0, 5)
        let x = getRndInteger(1, m)
        for (let j = 0; j < m+1; j++) {
            if (j === x) {
                candidates.push(characters[i])
            }
            else {
                candidates.push(getRndChar())
            }
        }
    }
    console.log(characters)
    console.log(candidates)
    console.log(needleman_wunsch_align(characters, candidates))
}

run_test()
