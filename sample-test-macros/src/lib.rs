extern crate proc_macro;
extern crate proc_macro2;
extern crate quote;
extern crate syn;

use std::mem;

use proc_macro::TokenStream;
use quote::{format_ident, quote};
use syn::{
    parse::{Parse, Parser},
    parse_quote,
    spanned::Spanned,
};

#[proc_macro_attribute]
pub fn sample_test(_args: TokenStream, input: TokenStream) -> TokenStream {
    let output = match syn::Item::parse.parse(input.clone()) {
        Ok(syn::Item::Fn(mut item_fn)) => {
            let mut inputs = syn::punctuated::Punctuated::new();
            let mut generators = Vec::new();
            let mut unwraps = Vec::new();
            let mut errors = Vec::new();

            item_fn
                .sig
                .inputs
                .iter_mut()
                .for_each(|input| match *input {
                    syn::FnArg::Typed(syn::PatType {
                        ref pat,
                        ref mut ty,
                        ref mut attrs,
                        ..
                    }) => {
                        let ix = attrs
                            .iter()
                            .position(|a| a.path.segments.iter().map(|s| &s.ident).eq(["sample"]));
                        if let Some(ix) = ix {
                            let id = format_ident!("_Sampletest{}", generators.len());
                            generators.push((id.clone(), ty.clone(), attrs.remove(ix)));
                            inputs.push(parse_quote!(_: #id));
                            *ty = parse_quote!(#id);
                            let unwrap_stmt: syn::Stmt = parse_quote!(let #pat = #pat.0;);
                            unwraps.push(unwrap_stmt);
                        } else {
                            inputs.push(parse_quote!(_: #ty));
                        }
                    }
                    _ => errors.push(syn::parse::Error::new(
                        input.span(),
                        "unsupported kind of function argument",
                    )),
                });

            if errors.is_empty() {
                let gen_impls = generators.iter().map(|(ty_id, ty, attr)| {
                    let literal = &attr.tokens;

                    quote! {
                        #[derive(Clone, Debug)]
                        struct #ty_id(#ty);

                        impl ::quickcheck::Arbitrary for #ty_id {
                            fn arbitrary(g: &mut ::quickcheck::Gen) -> Self {
                                let mut random = ::sample_std::Random::new(g.size());
                                #ty_id(::sample_std::Sample::generate(
                                    ::std::ops::Deref::deref(&(#literal)),
                                    &mut random,
                                ))
                            }

                            fn shrink(&self) -> Box<dyn Iterator<Item = Self>> {
                                Box::new(::sample_std::Sample::shrink(
                                    ::std::ops::Deref::deref(&(#literal)),
                                    self.0.clone()
                                ).map(|v| #ty_id(v)))
                            }
                        }
                    }
                });

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

                item_fn.block.stmts = unwraps
                    .into_iter()
                    .chain(item_fn.block.stmts.iter().cloned())
                    .collect();

                quote! {
                    #[test]
                    #(#attrs)*
                    fn #name() {
                        #(#gen_impls)*

                        #item_fn
                        ::quickcheck::quickcheck(#name as #fn_type)
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
            let msg = "#[quickcheck] is only supported on statics and functions";

            syn::parse::Error::new(span, msg).to_compile_error()
        }
    };

    output.into()
}

#[cfg(test)]
mod tests {}
