extern crate proc_macro;
extern crate proc_macro2;
extern crate quote;
extern crate syn;

use std::mem;

use proc_macro::TokenStream;
use quote::quote;
use syn::{
    parse::{Parse, Parser},
    spanned::Spanned,
};

#[proc_macro_attribute]
pub fn sample_test(_args: TokenStream, input: TokenStream) -> TokenStream {
    let output = match syn::Item::parse.parse(input.clone()) {
        Ok(syn::Item::Fn(mut item_fn)) => {
            let mut inputs = syn::punctuated::Punctuated::new();
            let mut samplers: syn::punctuated::Punctuated<_, syn::token::Comma> =
                syn::punctuated::Punctuated::new();
            let mut errors = Vec::new();

            item_fn
                .sig
                .inputs
                .iter_mut()
                .for_each(|input| match *input {
                    syn::FnArg::Typed(syn::PatType {
                        ref mut ty,
                        ref mut attrs,
                        ..
                    }) => {
                        let ix = attrs
                            .iter()
                            .position(|a| a.path.segments.iter().map(|s| &s.ident).eq(["sample"]));
                        if let Some(ix) = ix {
                            samplers.push(attrs.remove(ix).tokens);
                            inputs.push(syn::BareFnArg {
                                attrs: attrs.clone(),
                                name: None,
                                ty: *ty.clone(),
                            })
                        }
                    }
                    _ => errors.push(syn::parse::Error::new(
                        input.span(),
                        "unsupported kind of function argument",
                    )),
                });

            if errors.is_empty() {
                let attrs = mem::replace(&mut item_fn.attrs, Vec::new());
                let name = &item_fn.sig.ident;
                let fn_type = syn::TypeBareFn {
                    lifetimes: None,
                    unsafety: item_fn.sig.unsafety.clone(),
                    abi: item_fn.sig.abi.clone(),
                    fn_token: <syn::Token![fn]>::default(),
                    paren_token: syn::token::Paren::default(),
                    inputs,
                    variadic: item_fn.sig.variadic.clone(),
                    output: item_fn.sig.output.clone(),
                };

                let x = samplers.clone();
                quote! {
                    #[test]
                    #(#attrs)*
                    fn #name() {
                        let sampler = (#x,);

                        #item_fn
                        ::sample_test::tester::sample_test(sampler, #name as #fn_type)
                    }
                }
            } else {
                errors
                    .iter()
                    .map(syn::parse::Error::to_compile_error)
                    .collect()
            }
        }
        Ok(syn::Item::Static(mut item_static)) => {
            let attrs = mem::replace(&mut item_static.attrs, Vec::new());
            let name = &item_static.ident;

            quote! {
                #[test]
                #(#attrs)*
                fn #name() {
                    #item_static
                    ::quickcheck::quickcheck(#name)
                }
            }
        }
        _ => {
            let span = proc_macro2::TokenStream::from(input).span();
            let msg = "#[sample_test] is only supported on statics and functions";

            syn::parse::Error::new(span, msg).to_compile_error()
        }
    };

    output.into()
}

#[cfg(test)]
mod tests {}
