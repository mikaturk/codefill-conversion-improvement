import { readdir, readFile, writeFile } from "node:fs/promises"
import { join } from "node:path"
import { convert } from "./convert.js";

const JS_FOLDER = "/mnt/mturk/cf_sample_data/javascript/"
const JS_SRC_FOLDER = join(JS_FOLDER, "js-dataset")
const JS_CONVERTED_FOLDER = join(JS_FOLDER, "converted")

const files = await readdir(JS_SRC_FOLDER)

async function convert_file(path, output_path) {
    const source_str = await readFile(path).then(x=>x.toString())
    const converted = convert(source_str);
    await writeFile(output_path, converted);
}

// Single threaded loop
// TODO: Add multithreading
for (const file of files) {
    const path = join(JS_SRC_FOLDER, file);
    const converted_path = join(JS_CONVERTED_FOLDER, file + ".txt");
    convert_file(path, converted_path);
}


