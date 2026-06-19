/**
 * Lightweight BERT / WordPiece tokenizer for all-MiniLM-L6-v2 routing.
 *
 * Loads models/vocab.txt and produces fixed-length [CLS] + tokens + [SEP] + 
 * sequences for ONNX inference.
 */

#pragma once

#include <algorithm>
#include <cctype>
#include <cstdint>
#include <fstream>
#include <sstream>
#include <stdexcept>
#include <string>
#include <unordered_map>
#include <vector>

class BertWordPieceTokenizer {
public:
    static constexpr int64_t kPadId = 0;
    static constexpr int64_t kUnkId = 100;
    static constexpr int64_t kClsId = 101;
    static constexpr int64_t kSepId = 102;

    explicit BertWordPieceTokenizer(std::string vocab_path)
        : vocab_path_(std::move(vocab_path)) {
        load_vocab(vocab_path_);
    }

    /** Tokenize text into exactly max_len ids: [CLS] + wordpieces + [SEP] + . */
    std::vector<int64_t> tokenize(const std::string& text, int max_len = 16) const {
        if (max_len < 2) {
            throw std::invalid_argument("max_len must be at least 2 for [CLS] and [SEP]");
        }

        std::vector<int64_t> ids;
        ids.reserve(static_cast<std::size_t>(max_len));
        ids.push_back(kClsId);

        const int max_content = max_len - 2;  // reserve CLS + SEP
        auto basic_tokens = basic_tokenize(to_lower(text));

        for (const auto& token : basic_tokens) {
            if (static_cast<int>(ids.size()) - 1 >= max_content) {
                break;
            }
            auto pieces = wordpiece_tokenize(token);
            for (const auto& piece : pieces) {
                if (static_cast<int>(ids.size()) - 1 >= max_content) {
                    break;
                }
                ids.push_back(lookup(piece));
            }
        }

        ids.push_back(kSepId);

        while (static_cast<int>(ids.size()) < max_len) {
            ids.push_back(kPadId);
        }

        return ids;
    }

    /** Attention mask: 1 for real tokens, 0 for padding. */
    std::vector<int64_t> attention_mask(const std::vector<int64_t>& input_ids) const {
        std::vector<int64_t> mask;
        mask.reserve(input_ids.size());
        for (int64_t id : input_ids) {
            mask.push_back(id == kPadId ? 0 : 1);
        }
        return mask;
    }

    /** Single-segment BERT inputs use all-zero token type ids. */
    std::vector<int64_t> token_type_ids(std::size_t length) const {
        return std::vector<int64_t>(length, 0);
    }

    const std::string& vocab_path() const { return vocab_path_; }

private:
    std::string vocab_path_;
    std::unordered_map<std::string, int64_t> vocab_;

    void load_vocab(const std::string& path) {
        std::ifstream file(path);
        if (!file) {
            throw std::runtime_error("Failed to open vocab file: " + path);
        }

        std::string line;
        int64_t index = 0;
        while (std::getline(file, line)) {
            if (!line.empty() && line.back() == '\r') {
                line.pop_back();
            }
            vocab_[line] = index++;
        }

        if (vocab_.empty()) {
            throw std::runtime_error("Vocab file is empty: " + path);
        }
    }

    static std::string to_lower(std::string text) {
        std::transform(text.begin(), text.end(), text.begin(), [](unsigned char c) {
            return static_cast<char>(std::tolower(c));
        });
        return text;
    }

    static bool is_punctuation(char c) {
        return (c >= 33 && c <= 47) || (c >= 58 && c <= 64) ||
               (c >= 91 && c <= 96) || (c >= 123 && c <= 126);
    }

    static std::vector<std::string> whitespace_tokenize(const std::string& text) {
        std::vector<std::string> tokens;
        std::istringstream stream(text);
        std::string word;
        while (stream >> word) {
            tokens.push_back(word);
        }
        return tokens;
    }

    static std::vector<std::string> split_on_punctuation(const std::string& token) {
        std::vector<std::string> output;
        std::string current;
        for (char c : token) {
            if (is_punctuation(c)) {
                if (!current.empty()) {
                    output.push_back(current);
                    current.clear();
                }
                output.push_back(std::string(1, c));
            } else {
                current += c;
            }
        }
        if (!current.empty()) {
            output.push_back(current);
        }
        return output;
    }

    std::vector<std::string> basic_tokenize(const std::string& text) const {
        std::vector<std::string> tokens;
        for (const auto& token : whitespace_tokenize(text)) {
            auto parts = split_on_punctuation(token);
            tokens.insert(tokens.end(), parts.begin(), parts.end());
        }
        return tokens;
    }

    std::vector<std::string> wordpiece_tokenize(const std::string& word) const {
        if (word.empty()) {
            return {};
        }

        if (vocab_.find(word) != vocab_.end()) {
            return {word};
        }

        std::vector<std::string> subwords;
        std::size_t start = 0;
        const std::size_t length = word.size();

        while (start < length) {
            std::size_t end = length;
            std::string cur_substr;
            bool found = false;

            while (start < end) {
                std::string substr = word.substr(start, end - start);
                if (start > 0) {
                    substr = "##" + substr;
                }
                if (vocab_.find(substr) != vocab_.end()) {
                    cur_substr = substr;
                    found = true;
                    break;
                }
                --end;
            }

            if (!found) {
                subwords.push_back("[UNK]");
                break;
            }

            subwords.push_back(cur_substr);
            start = end;
        }

        return subwords;
    }

    int64_t lookup(const std::string& token) const {
        auto it = vocab_.find(token);
        if (it != vocab_.end()) {
            return it->second;
        }
        return kUnkId;
    }
};
